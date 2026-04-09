---
name: Device Health Check
triggers: device down, device unreachable, device health, device check, high CPU, check device, what is wrong with
---

Use this runbook when investigating the overall health of a specific device.

## What to investigate

1. Get all interfaces on the device. Identify which are down (admin or link) and which are in an error state.

2. Check OSPF neighbors. Missing adjacencies indicate routing protocol issues.

3. Pull syslog. Look for interface flaps, OSPF changes, and any error messages.

4. Check interface counters on any interfaces that look problematic.

5. Search ServiceNow for recent incidents and changes on this device.

## What to look for

- Multiple interfaces down simultaneously (power, upstream, or config issue)
- OSPF adjacencies missing
- Repeated syslog events indicating instability
- Recent changes in ServiceNow that correlate with the issue onset
