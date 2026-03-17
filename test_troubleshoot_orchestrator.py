"""
Smoke test for the troubleshoot orchestrator.
Patches _call_agent in troubleshoot_orchestrator to return fake agent responses,
so no NetBrain, Cisco, Panorama, or Splunk services are required.

The Ollama LLM must be running — this tests that the orchestrator LLM correctly
reasons over the fake agent responses and produces a structured troubleshooting report.

Run from repo root:
    .venv/bin/python test_troubleshoot_orchestrator.py
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Fake agent responses
# ---------------------------------------------------------------------------

FAKE_NETBRAIN_RESPONSE = """
Path from 10.0.0.1 to 10.0.1.1:
1. core-sw-01 (Switch) — Ethernet1/1 → Ethernet1/2
2. PA-FW-LEANDER (Palo Alto Firewall) — Ethernet1/1 → Ethernet1/2
3. dist-sw-01 (Switch) — Ethernet1/3 → Ethernet1/4

Path status: traffic DENIED at PA-FW-LEANDER. No matching permit policy found for source 10.0.0.1 destination 10.0.1.1 on TCP 443.
""".strip()

FAKE_PANORAMA_RESPONSE = """
PA-FW-LEANDER is in device group leander.
Ethernet1/1: trust zone, Ethernet1/2: untrust zone.

Security policies for 10.0.0.1 → 10.0.1.1 on TCP 443:
- Policy "deny-untrusted" (action: deny, source: any, destination: any, port: any) — matches this traffic.
- No permit policy found for this source/destination/port combination.

Address group membership: 10.0.0.1 is not in any address group in device group leander.
""".strip()

FAKE_SPLUNK_RESPONSE = """
Splunk analysis for 10.0.0.1 (last 24h):
- Deny events: 47 (all denied at PA-FW-LEANDER)
- Traffic summary: 47 deny, 0 allow
- Destination spread: 1 unique destination IP (10.0.1.1), 1 unique port (443)
- Pattern: repeated TCP 443 attempts to 10.0.1.1, all denied — consistent with a missing permit rule.
""".strip()


def fake_call_agent_factory():
    """Returns an async mock that dispatches fake responses by agent URL."""
    async def fake_call_agent(url: str, task: str, timeout: float = 180.0) -> str:
        if "8004" in url:   # NetBrain
            print(f"  [mock] NetBrain agent called: {task[:80]}...")
            return FAKE_NETBRAIN_RESPONSE
        if "8003" in url:   # Panorama
            print(f"  [mock] Panorama agent called: {task[:80]}...")
            return FAKE_PANORAMA_RESPONSE
        if "8002" in url:   # Splunk
            print(f"  [mock] Splunk agent called: {task[:80]}...")
            return FAKE_SPLUNK_RESPONSE
        return f"Agent at {url} unavailable (no mock defined)."
    return fake_call_agent


async def main():
    from atlas.agents.troubleshoot_orchestrator import orchestrate_troubleshoot

    print("Troubleshoot orchestrator smoke test")
    print("All agent HTTP calls are mocked — only Ollama LLM is required")
    print("-" * 60)

    with patch(
        "atlas.agents.troubleshoot_orchestrator._call_agent",
        side_effect=fake_call_agent_factory(),
    ):
        result = await orchestrate_troubleshoot(
            prompt="Troubleshoot why 10.0.0.1 cannot reach 10.0.1.1 on TCP 443",
        )

    print("\nOrchestrator response:")
    content = result.get("content", {})
    if isinstance(content, dict):
        print(content.get("direct_answer", content))
    else:
        print(content)


if __name__ == "__main__":
    asyncio.run(main())
