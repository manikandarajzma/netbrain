"""
Runbook executor for Atlas troubleshooting.

Loads a YAML runbook and executes steps against a registry of async tool
callables, resolving {{variable}} templates from the shared context between
steps.

Supported step types
--------------------
Tool step:
    - id: my_step
      tool: some_agent
      input: {key: "{{var}}"}

Parallel block (steps run concurrently):
    - id: gather
      parallel:
        - id: snow
          tool: servicenow_agent ...
        - id: counters
          tool: interface_counters_agent ...

If/then/else branch:
    - id: check_ping
      if: ping_failed               # expression — see _eval_expr below
      then:
        - id: routing_check ...
      else:
        - id: note
          message: "Ping passed."

Message step (injects a note into outputs, no tool call):
    - id: note
      message: "No routing issues — path is healthy."

Condition expressions
---------------------
  - bare key:             "ping_failed"       → truthy ctx["ping_failed"]
  - logical operators:    "ping_failed and has_firewalls"
  - negation:             "not ping_failed"
  - comparison:           "ping_loss_pct > 50"
  - parentheses:          "(ping_failed or high_latency) and path_devices"

All names are resolved from the runbook context dict.
"""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable

import yaml

logger = logging.getLogger("atlas.runbook")


# ---------------------------------------------------------------------------
# Template resolution
# ---------------------------------------------------------------------------

def _resolve(value: Any, ctx: dict) -> Any:
    """Recursively resolve {{var}} templates in strings, lists, and dicts.

    A string that is *only* a {{token}} returns the raw context value so that
    lists and dicts pass through without being stringified.
    """
    if isinstance(value, str):
        single = re.fullmatch(r"\{\{(\w+)\}\}", value.strip())
        if single:
            return ctx.get(single.group(1), "")
        return re.sub(
            r"\{\{(\w+)\}\}",
            lambda m: str(ctx.get(m.group(1), "")),
            value,
        )
    if isinstance(value, dict):
        return {k: _resolve(v, ctx) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(item, ctx) for item in value]
    return value


# ---------------------------------------------------------------------------
# Condition / expression evaluation
# ---------------------------------------------------------------------------

# Tokens allowed in condition expressions — prevents arbitrary code execution.
_SAFE_TOKENS = re.compile(
    r'^[\w\s\(\)\.\'"<>=!,+-]+$'
)


def _eval_expr(expr: Any, ctx: dict) -> bool:
    """Evaluate a condition expression against the runbook context.

    Supported forms:
      - None / True        → always run
      - False              → always skip
      - "key"              → truthy ctx["key"]
      - "a and b"          → both truthy
      - "a or b"           → either truthy
      - "not a"            → falsy ctx["a"]
      - "ping_loss_pct > 50" → numeric comparison
      - "(a or b) and c"   → grouped
    """
    if expr is None or expr is True:
        return True
    if expr is False:
        return False
    if not isinstance(expr, str):
        return bool(expr)

    expr = expr.strip()

    # Simple bare key — fast path
    if re.fullmatch(r'\w+', expr):
        return bool(ctx.get(expr))

    # Safety check — only allow safe characters
    if not _SAFE_TOKENS.match(expr):
        logger.warning("runbook: unsafe condition expression %r — treating as False", expr)
        return False

    # Build a safe namespace: all ctx keys + python builtins we want to allow
    namespace = {k: v for k, v in ctx.items()}
    namespace["__builtins__"] = {}

    try:
        return bool(eval(expr, namespace))  # noqa: S307 — namespace is restricted
    except Exception as exc:
        logger.warning("runbook: condition %r eval failed: %s — treating as False", expr, exc)
        return False


# ---------------------------------------------------------------------------
# Dict → readable text for synthesis
# ---------------------------------------------------------------------------

def _dict_to_text(label: str, d: dict) -> str:
    return f"[{label}]\n{json.dumps(d, indent=2)}"


# ---------------------------------------------------------------------------
# Status push helper
# ---------------------------------------------------------------------------

# Human-readable status labels shown in the UI while each tool runs.
_TOOL_STATUS: dict[str, str] = {
    "path_agent":               "Tracing network path...",
    "reverse_path_agent":       "Tracing return path...",
    "servicenow_agent":         "Checking ServiceNow for incidents and changes...",
    "interface_counters_agent": "Polling interface error counters...",
    "ping_agent":               "Pinging from first-hop device...",
    "routing_check_agent":      "Checking routing table on each hop...",
    "panorama_agent":           "Checking Panorama firewall policies...",
    "splunk_agent":             "Querying Splunk for traffic patterns...",
    "tcp_port_agent":           "Testing TCP port reachability from last-hop device...",
    "ospf_agent":               "Checking OSPF neighbor state...",
    "ospf_interfaces_agent":    "Checking OSPF interface config...",
    "ospf_history_agent":       "Comparing OSPF neighbor history...",
    "routing_history_agent":    "Looking up routing history...",
}


def _build_status(tool_name: str, resolved: dict) -> str:
    """Build a human-readable status message using resolved input values (e.g. actual hostnames)."""
    device = resolved.get("device", "")
    devices = resolved.get("devices", "")
    destination = resolved.get("destination", "")
    port = resolved.get("port", "")

    if tool_name == "ping_agent":
        if device and destination:
            return f"Pinging from {device} → {destination}..."
        return "Pinging from first-hop device..."
    if tool_name == "tcp_port_agent":
        if device and destination and port:
            return f"Testing TCP {port} from {device} → {destination}..."
        return "Testing TCP port reachability..."
    if tool_name == "routing_check_agent":
        if devices:
            return f"Checking routing on: {devices}..."
        return "Checking routing table on each hop..."
    if tool_name == "path_agent":
        return "Tracing network path via live SSH..."
    return _TOOL_STATUS.get(tool_name, f"Running {tool_name}...")


async def _push_status(session_id: str, message: str) -> None:
    try:
        try:
            import atlas.status_bus as status_bus
        except ImportError:
            import status_bus  # type: ignore
        await status_bus.push(session_id, message)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# RunbookExecutor
# ---------------------------------------------------------------------------

class RunbookExecutor:
    """
    Execute a YAML runbook against a tool registry.

    Parameters
    ----------
    runbook_path:    Path to the YAML runbook file.
    tools:           Dict of {tool_name: callable}.  Each callable may be async
                     or sync; the executor handles both.
    post_processors: Dict of {step_id: fn(result, ctx)}.  Called after each
                     step to extract structured fields into the shared context.
                     May be async.
    session_id:      Used to push live status messages to the UI.
    """

    def __init__(
        self,
        runbook_path: str | Path,
        tools: dict[str, Callable],
        post_processors: dict[str, Callable] | None = None,
        session_id: str = "default",
    ):
        path = Path(runbook_path)
        with open(path) as f:
            self.spec = yaml.safe_load(f)
        self.tools = tools
        self.post_processors = post_processors or {}
        self.session_id = session_id
        self.ctx: dict[str, Any] = {}
        self.outputs: list[str] = []

    async def run(self, initial_context: dict) -> list[str]:
        """Execute all runbook steps and return text outputs for synthesis."""
        self.ctx.update(initial_context)
        self.outputs = []
        runbook_name = self.spec.get("name", "runbook")
        await _push_status(self.session_id, f"Running {runbook_name} analysis...")
        for step in self.spec.get("steps", []):
            await self._run_step(step)
        return self.outputs

    async def _run_steps(self, steps: list[dict], _in_parallel: bool = False) -> None:
        """Run a list of steps sequentially."""
        for step in steps:
            await self._run_step(step, _in_parallel=_in_parallel)

    async def _run_step(self, step: dict, _in_parallel: bool = False) -> None:
        step_id = step.get("id", "_anon")

        # ── If/then/else branch ────────────────────────────────────────────
        if "if" in step:
            condition = step["if"]
            branch = "then" if _eval_expr(condition, self.ctx) else "else"
            branch_steps = step.get(branch, [])
            logger.info("runbook[%s]: if=%r → %s (%d steps)", step_id, condition, branch, len(branch_steps))
            if branch_steps:
                await _push_status(
                    self.session_id,
                    f"{'✓' if branch == 'then' else '↷'} {step_id}: {branch} branch",
                )
                await self._run_steps(branch_steps, _in_parallel=_in_parallel)
            return

        # ── Message step (no tool — injects a note into outputs) ──────────
        if "message" in step:
            msg = _resolve(step["message"], self.ctx)
            logger.info("runbook[%s]: message=%r", step_id, str(msg)[:80])
            self.outputs.append(str(msg))
            return

        # ── Parallel block ─────────────────────────────────────────────────
        if "parallel" in step:
            labels = []
            for s in step["parallel"]:
                t = s.get("tool", "")
                if t:
                    resolved_preview = _resolve(s.get("input", {}), self.ctx)
                    labels.append(_build_status(t, resolved_preview))
            if labels:
                await _push_status(self.session_id, " | ".join(labels))
            await asyncio.gather(*[self._run_step(s, _in_parallel=True) for s in step["parallel"]])
            return

        # ── Single tool step ───────────────────────────────────────────────
        tool_name = step.get("tool")
        if not tool_name:
            return

        fn = self.tools.get(tool_name)
        if not fn:
            logger.warning("runbook[%s]: no tool registered for '%s'", step_id, tool_name)
            return

        resolved = _resolve(step.get("input", {}), self.ctx)

        # Push individual status only when NOT inside a parallel block
        if not _in_parallel:
            status_msg = _build_status(tool_name, resolved)
            await _push_status(self.session_id, status_msg)
        logger.info(
            "runbook[%s]: %s(%s)",
            step_id, tool_name,
            {k: str(v)[:80] for k, v in resolved.items()},
        )

        try:
            if asyncio.iscoroutinefunction(fn):
                result = await fn(**resolved)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: fn(**resolved))
        except Exception as exc:
            logger.error("runbook[%s]: %s raised %s", step_id, tool_name, exc)
            result = f"[{tool_name} failed: {exc}]"

        # Store result in context under this step's id
        self.ctx[step_id] = result

        # Post-processor (extracts structured fields into ctx); may be async
        post = self.post_processors.get(step_id)
        if post:
            try:
                if asyncio.iscoroutinefunction(post):
                    await post(result, self.ctx)
                else:
                    post(result, self.ctx)
            except Exception as exc:
                logger.warning("runbook[%s]: post-processor failed: %s", step_id, exc)

        # Accumulate text output for the synthesis LLM
        if isinstance(result, str) and result.strip():
            self.outputs.append(result)
        elif isinstance(result, dict):
            self.outputs.append(_dict_to_text(step_id, result))
