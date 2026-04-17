# Atlas LLM Tool Instruction Flow

This document explains how Atlas tells the LLM what tools it can use, how the LLM is guided to use them, and which parts are still enforced by code.

## 1. The LLM role is chosen first

File: `/Users/manig/Documents/coding/atlas/agents/agent_factory.py`

```python
class AgentFactory:
    def __init__(self) -> None:
        self.router_model = os.getenv("ATLAS_ROUTER_MODEL", "gemma4:latest")
        self.selector_model = os.getenv("ATLAS_SELECTOR_MODEL", "gemma4:latest")
        self.network_ops_model = os.getenv("ATLAS_NETWORK_OPS_MODEL", "gemma4:latest")
        self.troubleshoot_model = os.getenv("ATLAS_TROUBLESHOOT_MODEL", "gemma4:latest")
```

Atlas chooses a model role before any agent run starts:

- router model
- selector model
- network-ops agent model
- troubleshoot agent model

## 2. The first LLM decides the top-level lane

Files:

- `/Users/manig/Documents/coding/atlas/graph/graph_nodes.py`
- `/Users/manig/Documents/coding/atlas/services/intent_routing_service.py`

```python
llm_decision = await intent_routing_service.route_prompt(prompt)
if llm_decision:
    llm_intent = str(llm_decision.get("intent") or "").strip()
    return {"intent": llm_intent}
```

```python
_ROUTER_SYSTEM_PROMPT = """You are Atlas's coarse request router.

Classify the user's request into exactly one intent:
- troubleshoot
- network_ops
- dismiss
"""
```

```python
llm = agent_factory.build_router_llm()
response = await llm.ainvoke(
    [
        SystemMessage(content=_ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
)
```

The LLM is asked to choose exactly one of:

- `troubleshoot`
- `network_ops`
- `dismiss`

## 3. A second LLM chooses the scenario

After the lane is chosen, a selector LLM picks the scenario.

### Troubleshooting scenarios

File: `/Users/manig/Documents/coding/atlas/services/troubleshoot_scenario_service.py`

```python
_SCENARIO_SYSTEM_PROMPT = """You are Atlas's troubleshooting scenario selector.

Choose exactly one scenario:
- connectivity
- performance
- intermittent
- general
"""
```

### Network-ops scenarios

File: `/Users/manig/Documents/coding/atlas/services/network_ops_scenario_service.py`

```python
_SCENARIO_SYSTEM_PROMPT = """You are Atlas's network-operations scenario selector.

Choose exactly one scenario:
- incident_record
- record_lookup
- change_record
- change_update
- access_change
- general
"""
```

Both selectors use the selector model:

```python
llm = agent_factory.build_selector_llm()
```

## 4. Code decides which tools are even visible

File: `/Users/manig/Documents/coding/atlas/tools/tool_registry.py`

The LLM does not automatically see every tool in the repo. Code decides visibility in three steps.

### 4A. Tools are tagged with capabilities

```python
PATH_TOOL_CAPABILITIES = (
    (trace_path, ("workflow.path.trace",)),
    (trace_reverse_path, ("workflow.path.reverse_trace",)),
)
```

Source: `/Users/manig/Documents/coding/atlas/tools/path_agent_tools.py`

### 4B. Agent profiles define allowed capabilities

```python
self.register_profile(
    "troubleshoot.connectivity",
    (
        "workflow.path.trace",
        "workflow.path.reverse_trace",
        "workflow.connectivity.ping",
        "workflow.routing.check",
        "workflow.connectivity.snapshot",
        "workflow.routing.history",
        "servicenow.search",
        "servicenow.incident.read",
        "memory.recall",
        "knowledge.vendor.lookup",
    ),
)
```

```python
self.register_profile(
    "network_ops",
    (
        "workflow.path.trace",
        "servicenow.search",
        "servicenow.incident.read",
        "servicenow.change.read",
        "servicenow.incident.create",
        "servicenow.change.create",
        "servicenow.change.update",
    ),
)
```

### 4C. ToolRegistry resolves the profile into the final tool list

```python
def get_profile_tools(self, profile_name: ProfileName) -> tuple[Any, ...]:
    capabilities = self._profiles.get(profile_name)
    tools = self.get_tools_for_capabilities(capabilities)
    return tools
```

In plain English:

1. Each tool is tagged with capabilities.
2. Each agent profile is assigned a capability set.
3. ToolRegistry resolves that profile into the final visible tool list.
4. The agent only receives that tool list.

## 5. The agent builder loads the prompt and the allowed tools

### Troubleshoot agent

File: `/Users/manig/Documents/coding/atlas/agents/troubleshoot_agent.py`

```python
ALL_TOOLS = tool_registry.get_profile_tools("troubleshoot.general")
CONNECTIVITY_TOOLS = tool_registry.get_profile_tools("troubleshoot.connectivity")

def build_agent(prompt: str = "", scenario: str = "general", *, llm=None):
    llm = llm or agent_factory.build_troubleshoot_llm()
    system_prompt = load_system_prompt(scenario)
    tools = CONNECTIVITY_TOOLS if scenario_path.endswith("connectivity.md") else ALL_TOOLS
    return agent_factory.create_specialized_agent(llm, tools, system_prompt, "troubleshoot")
```

### Network-ops agent

File: `/Users/manig/Documents/coding/atlas/agents/network_ops_agent.py`

```python
NETWORK_OPS_TOOLS = tool_registry.get_profile_tools("network_ops")

def build_agent(prompt: str = "", scenario: str = "general", *, llm=None):
    llm = llm or agent_factory.build_network_ops_llm()
    return agent_factory.create_specialized_agent(
        llm,
        NETWORK_OPS_TOOLS,
        load_system_prompt(scenario),
        "network_ops",
    )
```

By this point, Atlas has already decided:

- which model role to use
- which scenario prompt to load
- which tools are visible

## 6. The system prompt tells the LLM how to use tools

### Troubleshooting prompt

File: `/Users/manig/Documents/coding/atlas/skills/troubleshooter.md`

```text
You are a network troubleshooting agent. Investigate network problems by calling tools, reasoning about findings, and writing a precise root cause analysis.

- Always call tools yourself.
- Only report what tools returned.
- Always call `search_servicenow`
- Treat the scenario-specific runbook as the source of truth
```

### Network-ops prompt

File: `/Users/manig/Documents/coding/atlas/skills/network_ops.md`

```text
For explicit requests to create/open/raise an incident or ticket:
- use `create_servicenow_incident(...)`
- if source and destination IPs are present, you may call `trace_path(...)`
- if useful, call `search_servicenow(...)`
```

This is prompt-level instruction about when and why tools should be used.

## 7. The scenario runbook gives more tool-specific guidance

File: `/Users/manig/Documents/coding/atlas/skills/troubleshooting_scenarios/connectivity.md`

```text
Step 1 — `trace_path(source_ip, dest_ip)` — always first.

Step 2 — in parallel:
- `trace_reverse_path(...)`
- `lookup_routing_history(...)`
- `search_servicenow(...)`

Step 3 — `collect_connectivity_snapshot(...)`
```

This is more detailed task-specific instruction layered on top of the core system prompt.

## 8. The tool itself also instructs the LLM

File: `/Users/manig/Documents/coding/atlas/tools/path_agent_tools.py`

```python
@tool
async def trace_path(source_ip: str, dest_ip: str, config: RunnableConfig) -> str:
    """
    Trace the hop-by-hop network path from source_ip to dest_ip via live SSH.
    Always call this FIRST for any connectivity troubleshooting query.
    ...
    The result tells you:
    - All device hostnames in the path
    - The first-hop device
    - The last-hop device
    """
```

The tool decorator plus the docstring gives the LLM:

- tool name
- arguments
- return shape
- usage guidance

## 9. The ReAct agent binds prompt and tools together

File: `/Users/manig/Documents/coding/atlas/agents/agent_factory.py`

```python
def create_specialized_agent(self, llm, tools, system_prompt: str, agent_name: str):
    return create_react_agent(
        llm,
        tools,
        prompt=SystemMessage(content=system_prompt),
        name=agent_name,
    )
```

This is the binding point. The LLM gets:

- the system prompt
- the scenario prompt
- the visible tool list
- the tool schemas and docstrings

## 10. The workflow invokes the agent with the user message

File: `/Users/manig/Documents/coding/atlas/services/troubleshoot_workflow_service.py`

```python
agent = build_agent(full_prompt, scenario)
result = await agent.ainvoke({"messages": [HumanMessage(content=agent_input)]}, config=config)
```

At that point the LLM can:

- read the user request
- choose one of the visible tools
- read the tool result
- choose another visible tool
- write the answer

## 11. Some safety-critical steps are still enforced by code

File: `/Users/manig/Documents/coding/atlas/services/troubleshoot_workflow_service.py`

```python
if workflow_state_service.needs_connectivity_snapshot(session_data, src_ip, dst_ip):
    follow_up = (
        ...
        "- Call collect_connectivity_snapshot(...) before writing the report.\n"
    )
    agent = build_agent(full_prompt, scenario)
    result = await agent.ainvoke(...)
```

```python
if workflow_state_service.missing_path_visuals(session_data, src_ip, dst_ip):
    await trace_path.ainvoke(...)
    await trace_reverse_path.ainvoke(...)
```

So the full truth is:

- The LLM is instructed through prompts and tool schemas.
- The LLM chooses many tool calls.
- Code still forces some safety-critical evidence steps.

## 12. The updated stack

```text
1. agent_factory.py chooses which model role is used
2. intent_routing_service.py uses an LLM to choose the lane
3. scenario service uses an LLM to choose the scenario
4. tool_registry.py decides which tools are visible
5. agent builder loads the core prompt + scenario prompt + visible tool list
6. tool docstrings describe what each tool does
7. create_react_agent(...) gives the LLM the prompt and tools
8. the LLM chooses tools inside that visible set
9. workflow code still forces safety-critical evidence when needed
```

## One-line summary

Atlas instructs the LLM to use tools through role-specific prompts, scenario runbooks, tool docstrings, and a bounded tool list from ToolRegistry; the LLM can only choose from the tools code exposes to it.
