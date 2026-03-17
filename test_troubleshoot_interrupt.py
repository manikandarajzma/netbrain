"""
Smoke test for the troubleshoot orchestrator interrupt/resume flow.

Tests two scenarios:
  1. Vague prompt → interrupt fires → user replies → investigation runs
  2. Specific prompt → no interrupt → investigation runs immediately

All agent HTTP calls are mocked. Only Ollama must be running.

Run from repo root:
    .venv/bin/python test_troubleshoot_interrupt.py
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FAKE_NETBRAIN_RESPONSE = (
    "Path from 10.0.0.1 to 10.0.1.1: core-sw-01 → PA-FW-LEANDER → dist-sw-01. "
    "Traffic DENIED at PA-FW-LEANDER on TCP 443."
)
FAKE_PANORAMA_RESPONSE = (
    "PA-FW-LEANDER (device group: leander). Ethernet1/1: trust, Ethernet1/2: untrust. "
    "Policy 'deny-untrusted' (action: deny) matches this traffic. No permit rule found."
)
FAKE_SPLUNK_RESPONSE = (
    "10.0.0.1 — 47 deny events (24h), all on TCP 443 to 10.0.1.1. "
    "0 allow events. Consistent with missing permit rule."
)


async def fake_call_agent(url: str, task: str, timeout: float = 180.0) -> str:
    if "8004" in url:
        print(f"    [mock] NetBrain ← {task[:70]}...")
        return FAKE_NETBRAIN_RESPONSE
    if "8003" in url:
        print(f"    [mock] Panorama ← {task[:70]}...")
        return FAKE_PANORAMA_RESPONSE
    if "8002" in url:
        print(f"    [mock] Splunk   ← {task[:70]}...")
        return FAKE_SPLUNK_RESPONSE
    return "Agent unavailable."


def _print_response(label: str, response: dict) -> None:
    print(f"\n{'─' * 60}")
    print(f"Atlas ({label}):")
    content = response.get("content", "")
    if isinstance(content, dict):
        print(content.get("direct_answer", content))
    else:
        print(content)


async def scenario_1_vague_prompt():
    """Vague prompt: expect interrupt → reply → investigation."""
    print("\n" + "═" * 60)
    print("SCENARIO 1 — Vague prompt (expect clarifying question)")
    print("═" * 60)

    from atlas.chat_service import process_message

    with patch("atlas.agents.troubleshoot_orchestrator._call_agent", side_effect=fake_call_agent):

        # Turn 1: vague — no issue type, no port
        print("\nUser: troubleshoot between 10.0.0.1 and 10.0.1.1")
        r1 = await process_message(
            prompt="troubleshoot between 10.0.0.1 and 10.0.1.1",
            conversation_history=[],
            session_id="test-session-vague",
        )
        _print_response("Turn 1 — should be a clarifying question", r1)

        content = r1.get("content", "")
        assert isinstance(content, str) and ("blocked" in content.lower() or "port" in content.lower()), \
            f"Expected a clarifying question, got: {content}"
        print("\n✓ Interrupt fired correctly")

        # Turn 2: user answers
        print("\nUser: traffic is blocked on TCP 443")
        r2 = await process_message(
            prompt="traffic is blocked on TCP 443",
            conversation_history=[],
            session_id="test-session-vague",
        )
        _print_response("Turn 2 — should be the troubleshooting report", r2)

        content2 = r2.get("content", {})
        assert isinstance(content2, dict) and content2.get("direct_answer"), \
            f"Expected a troubleshooting report, got: {content2}"
        print("\n✓ Investigation completed after clarification")


async def scenario_2_specific_prompt():
    """Specific prompt: no interrupt, goes straight to investigation."""
    print("\n" + "═" * 60)
    print("SCENARIO 2 — Specific prompt (expect direct investigation)")
    print("═" * 60)

    from atlas.chat_service import process_message

    with patch("atlas.agents.troubleshoot_orchestrator._call_agent", side_effect=fake_call_agent):

        print("\nUser: troubleshoot why 10.0.0.1 cannot reach 10.0.1.1 on TCP 443")
        r = await process_message(
            prompt="troubleshoot why 10.0.0.1 cannot reach 10.0.1.1 on TCP 443",
            conversation_history=[],
            session_id="test-session-specific",
        )
        _print_response("Turn 1 — should be the troubleshooting report", r)

        content = r.get("content", {})
        assert isinstance(content, dict) and content.get("direct_answer"), \
            f"Expected a troubleshooting report without interrupt, got: {content}"
        print("\n✓ No interrupt — investigation ran immediately")


async def main():
    print("Troubleshoot interrupt/resume smoke test")
    print("Requires: Ollama running. All agents are mocked.")

    await scenario_1_vague_prompt()
    await scenario_2_specific_prompt()

    print("\n" + "═" * 60)
    print("All scenarios passed.")


if __name__ == "__main__":
    asyncio.run(main())
