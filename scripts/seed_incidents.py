"""
Seed script: close all open incidents and create 5 test incidents
with realistic troubleshooting steps and resolution notes.
Run: uv run python scripts/seed_incidents.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import servicenowauth
from tools.servicenow_tools import _base_url, _auth
import aiohttp

INCIDENTS_TO_CREATE = [
    {
        "short_description": "Connectivity loss from 10.0.0.1 to 11.0.0.1 - traffic dropped at PA-FW-01",
        "description": (
            "Users in the 10.0.0.0/24 subnet are unable to reach hosts in 11.0.0.0/24. "
            "NetBrain path trace confirms traffic reaches PA-FW-01 but does not egress."
        ),
        "work_notes": (
            "Troubleshooting steps:\n"
            "1. Ran NetBrain path trace — path: EDGE-RTR-01 → CORE-SW-01 → PA-FW-01 → DIST-RTR-02\n"
            "2. Checked Panorama policy match on PA-FW-01 — rule 'Block-External' was shadowing 'Allow-Internal-to-DMZ'\n"
            "3. Verified Splunk deny events — 142 deny hits on PA-FW-01 in last 24h for src 10.0.0.1\n"
            "4. Confirmed rule ordering issue — 'Block-External' was positioned above 'Allow-Internal-to-DMZ'\n"
            "5. Reordered security rules in Panorama device group DataCenter-DG\n"
            "6. Committed and pushed policy — connectivity restored"
        ),
        "close_notes": (
            "Root cause: Firewall rule ordering issue on PA-FW-01. "
            "'Block-External' rule was shadowing 'Allow-Internal-to-DMZ' due to incorrect rule position. "
            "Fixed by reordering rules in Panorama DataCenter-DG device group and pushing policy."
        ),
        "cmdb_ci": "PA-FW-01",
    },
    {
        "short_description": "Intermittent packet loss between 10.0.0.1 and 11.0.0.1 via CORE-SW-01",
        "description": (
            "Network monitoring alerts showing 15-20% packet loss on the path between "
            "10.0.0.0/24 and 11.0.0.0/24. Issue is intermittent, worse during business hours."
        ),
        "work_notes": (
            "Troubleshooting steps:\n"
            "1. Checked interface counters on CORE-SW-01 — GigabitEthernet1/0/24 showing input errors\n"
            "2. Examined SFP module diagnostics — TX power degraded on uplink to PA-FW-01\n"
            "3. Reviewed CDP/LLDP neighbors — confirmed single uplink, no redundancy\n"
            "4. Checked error logs — CRC errors spiking every ~4 hours correlating with packet loss\n"
            "5. Replaced SFP module on GigabitEthernet1/0/24\n"
            "6. Monitored for 2 hours — error counters cleared, packet loss resolved"
        ),
        "close_notes": (
            "Root cause: Degraded SFP module on CORE-SW-01 GigabitEthernet1/0/24 uplink to PA-FW-01. "
            "TX power was below acceptable threshold causing intermittent CRC errors and packet loss. "
            "Resolved by replacing the faulty SFP module."
        ),
        "cmdb_ci": "CORE-SW-01",
    },
    {
        "short_description": "BGP route missing on EDGE-RTR-01 causing traffic blackhole to 11.0.0.0/24",
        "description": (
            "Traffic destined for 11.0.0.0/24 is being dropped at EDGE-RTR-01. "
            "Traceroutes from 10.0.0.1 time out at first hop."
        ),
        "work_notes": (
            "Troubleshooting steps:\n"
            "1. Checked routing table on EDGE-RTR-01 — no route to 11.0.0.0/24\n"
            "2. Verified BGP neighbor state — session to upstream provider was in Idle state\n"
            "3. Checked BGP logs — session had been reset 3 hours prior due to hold-timer expiry\n"
            "4. Investigated hold-timer — keepalive packets were being dropped by an ACL on GigabitEthernet0/0\n"
            "5. Found ACL 150 blocking TCP port 179 (BGP) inbound — recently added by network change CHG0029001\n"
            "6. Removed incorrect ACL entry — BGP session re-established within 30 seconds\n"
            "7. Verified 11.0.0.0/24 route repopulated in routing table"
        ),
        "close_notes": (
            "Root cause: BGP session on EDGE-RTR-01 was reset due to an ACL entry blocking TCP/179 (BGP keepalives). "
            "The ACL was incorrectly added during change CHG0029001. "
            "Removed the blocking ACL entry — BGP session recovered and routes to 11.0.0.0/24 restored."
        ),
        "cmdb_ci": "EDGE-RTR-01",
    },
    {
        "short_description": "PA-FW-01 CPU spike causing policy evaluation delays on all traffic",
        "description": (
            "PA-FW-01 CPU utilization spiking to 95%+ causing significant latency for all traffic "
            "passing through the firewall. Users reporting slow application response times."
        ),
        "work_notes": (
            "Troubleshooting steps:\n"
            "1. Checked PA-FW-01 system resources in Panorama — dataplane CPU at 96%\n"
            "2. Ran 'show running resource-monitor' — threat prevention engine consuming majority of CPU\n"
            "3. Checked threat logs — wildfire analysis submissions spiking from internal host 10.0.0.45\n"
            "4. Identified 10.0.0.45 was infected with malware generating high volumes of unusual traffic\n"
            "5. Isolated 10.0.0.45 by applying dynamic address group quarantine policy in Panorama\n"
            "6. Engaged endpoint team to remediate the infected host\n"
            "7. PA-FW-01 CPU returned to normal (12%) within 10 minutes of isolation"
        ),
        "close_notes": (
            "Root cause: Infected internal host 10.0.0.45 was generating high-volume anomalous traffic "
            "causing PA-FW-01 wildfire analysis engine to spike to 96% CPU utilization. "
            "Resolved by quarantining 10.0.0.45 via dynamic address group policy in Panorama. "
            "Endpoint team remediated the infected host."
        ),
        "cmdb_ci": "PA-FW-01",
    },
    {
        "short_description": "DIST-RTR-02 interface flapping causing route instability to 11.0.0.0/24",
        "description": (
            "DIST-RTR-02 GigabitEthernet0/3 interface is flapping every 2-3 minutes. "
            "Each flap triggers OSPF reconvergence causing ~30s outage for 11.0.0.0/24 hosts."
        ),
        "work_notes": (
            "Troubleshooting steps:\n"
            "1. Checked SYSLOG on DIST-RTR-02 — interface GigabitEthernet0/3 up/down messages every ~2 min\n"
            "2. Verified physical layer — cable and patch panel connections looked fine visually\n"
            "3. Checked interface error counters — no errors, but link speed negotiation failing\n"
            "4. Reviewed connected switch port — auto-negotiation mismatch between router (1G) and switch (100M forced)\n"
            "5. Checked change history — CHG0030033 had set switch port to 100M fixed speed\n"
            "6. Updated switch port to auto-negotiation — interface stabilized immediately\n"
            "7. Monitored OSPF adjacency — stable for 30 minutes, no further flaps"
        ),
        "close_notes": (
            "Root cause: Speed/duplex mismatch between DIST-RTR-02 GigabitEthernet0/3 (auto 1G) and "
            "the connected switch port (hard-coded 100M) introduced during CHG0030033. "
            "The mismatch caused repeated link renegotiation and interface flapping. "
            "Fixed by setting switch port back to auto-negotiation."
        ),
        "cmdb_ci": "DIST-RTR-02",
    },
]


async def close_open_incidents(session):
    print("Fetching open incidents...")
    params = {
        "sysparm_query": "state!=6^state!=7^state!=8",
        "sysparm_fields": "sys_id,number,short_description",
        "sysparm_limit": "100",
    }
    async with session.get(
        _base_url() + "/api/now/table/incident",
        params=params, auth=_auth(),
        timeout=aiohttp.ClientTimeout(total=30)
    ) as resp:
        data = await resp.json()

    records = data.get("result", [])
    print(f"Found {len(records)} open incidents to close")

    for rec in records:
        sys_id = rec["sys_id"]
        number = rec["number"]
        payload = {
            "state": "7",
            "close_code": "Solved (Permanently)",
            "close_notes": "Closing as part of test data cleanup.",
        }
        async with session.patch(
            _base_url() + f"/api/now/table/incident/{sys_id}",
            json=payload, auth=_auth(),
            timeout=aiohttp.ClientTimeout(total=15)
        ) as resp:
            if resp.status == 200:
                print(f"  Closed {number}")
            else:
                print(f"  Failed to close {number}: HTTP {resp.status}")


async def create_incident(session, inc: dict) -> str:
    payload = {
        "short_description": inc["short_description"],
        "description":       inc["description"],
        "urgency":           "2",
        "impact":            "2",
        "category":          "network",
    }
    async with session.post(
        _base_url() + "/api/now/table/incident",
        json=payload, auth=_auth(),
        timeout=aiohttp.ClientTimeout(total=15)
    ) as resp:
        data = await resp.json()
    result = data.get("result", {})
    sys_id = result.get("sys_id", "")
    number = result.get("number", "")
    if not sys_id:
        print(f"  Failed to create: {inc['short_description'][:60]}")
        return ""
    print(f"  Created {number}: {inc['short_description'][:60]}...")

    # Add work notes (troubleshooting steps)
    await session.patch(
        _base_url() + f"/api/now/table/incident/{sys_id}",
        json={"work_notes": inc["work_notes"]},
        auth=_auth(),
        timeout=aiohttp.ClientTimeout(total=15)
    )

    # Close with resolution notes
    close_payload = {
        "state":      "7",
        "close_code": "Solved (Permanently)",
        "close_notes": inc["close_notes"],
    }
    async with session.patch(
        _base_url() + f"/api/now/table/incident/{sys_id}",
        json=close_payload, auth=_auth(),
        timeout=aiohttp.ClientTimeout(total=15)
    ) as resp:
        if resp.status == 200:
            print(f"  Closed {number} with resolution notes")
        else:
            print(f"  Warning: close returned HTTP {resp.status} for {number}")
    return number


async def main():
    async with aiohttp.ClientSession() as session:
        await close_open_incidents(session)
        print(f"\nCreating {len(INCIDENTS_TO_CREATE)} incidents...")
        for inc in INCIDENTS_TO_CREATE:
            await create_incident(session, inc)
    print("\nDone. Run the memory sync to load these into RedisVL:")
    print("  redis-cli KEYS 'atlas:mem:inc:*' | wc -l  # should be 0 until sync runs")
    print("  Restart the backend to trigger the startup sync.")


asyncio.run(main())
