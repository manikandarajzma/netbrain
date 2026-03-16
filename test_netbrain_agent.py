"""
Smoke test for the NetBrain agent loop.
Patches netbrain_query_path.ainvoke to return a fake path with a PA firewall hop,
so the agent can be tested without a NetBrain license.

Requires the Panorama agent to be running on port 8003.

Run from repo root:
    .venv/bin/python test_netbrain_agent.py
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FAKE_PATH_RESULT = {
    "source": "10.0.0.1",
    "destination": "10.0.1.1",
    "path_status": "Succeeded",
    "path_hops": [
        {
            "device": "core-sw-01",
            "device_type": "Switch",
            "is_firewall": False,
            "in_interface": "Ethernet1/1",
            "out_interface": "Ethernet1/2",
        },
        {
            "device": "PA-FW-LEANDER",
            "device_type": "Palo Alto Firewall",
            "is_firewall": True,
            "firewall_device": "PA-FW-LEANDER",
            "in_interface": "Ethernet1/1",
            "out_interface": "Ethernet1/2",
        },
        {
            "device": "dist-sw-01",
            "device_type": "Switch",
            "is_firewall": False,
            "in_interface": "Ethernet1/3",
            "out_interface": "Ethernet1/4",
        },
    ],
}


async def fake_mcp(tool_name, params, **kwargs):
    if tool_name == "query_network_path":
        return FAKE_PATH_RESULT
    raise RuntimeError(f"Unexpected MCP call in test: {tool_name}")


async def main():
    from atlas.agents.agent_loop import run_agent_loop
    from atlas.agents.netbrain_agent import NETBRAIN_TOOLS, _load_skill

    print("Running NetBrain agent loop with fake path data...")
    print("Fake path includes PA firewall: PA-FW-LEANDER (Ethernet1/1 → Ethernet1/2)")
    print("Panorama agent must be running on port 8003")
    print("-" * 60)

    with patch("atlas.mcp_client.call_mcp_tool", side_effect=fake_mcp):
        result = await run_agent_loop(
            task="Find the network path from 10.0.0.1 to 10.0.1.1 and enrich any firewall hops with Panorama zone and device group information.",
            system_prompt=_load_skill(),
            tools=NETBRAIN_TOOLS,
        )

    print("\nAgent response:")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
