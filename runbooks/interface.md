---
name: Interface Troubleshooting
triggers: interface down, interface flapping, link down, port down, ethernet, CRC, errors, drops, flap, bouncing, err-disable
---

Use this runbook when a specific interface or port is reported down, flapping, or showing errors.

## What to investigate

1. Get the interface detail (oper status, line protocol, error counters, speed/duplex).

2. Pull syslog from the device. Look for LINEPROTO-5-UPDOWN events to find when it went down and how often it is flapping.

3. Check interface counters over time (3 polls). Actively incrementing CRC errors point to a physical layer problem (cable, SFP, or duplex mismatch). Incrementing input drops point to congestion.

4. Search ServiceNow for incidents and changes on this device. A recent change may have misconfigured the interface.

5. If the interface is a routing interface (has an IP), check OSPF neighbors and routing history to understand the downstream impact.

## What to look for

- Admin shutdown vs link-down (different root causes)
- CRC errors → physical problem (bad cable, SFP, or duplex mismatch)
- Input drops → congestion or policer
- Frequent flaps in syslog → unstable physical link or SFP
- Recent config changes in ServiceNow
