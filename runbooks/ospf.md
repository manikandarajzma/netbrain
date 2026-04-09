---
name: OSPF Troubleshooting
triggers: ospf, adjacency, routing protocol, neighbor, convergence, route missing, route changed, BGP, routing instability
---

Use this runbook when OSPF adjacencies are down, routes are missing, or routing has changed unexpectedly.

## What to investigate

1. Check OSPF neighbors on all relevant devices. Note which adjacencies are missing or in a non-FULL state.

2. Check which interfaces have OSPF enabled. Zero OSPF interfaces on a device means the process is misconfigured or the network statement is missing.

3. Pull OSPF neighbor count history from the database to see when adjacencies dropped and whether this is recurring.

4. Pull syslog from affected devices. Look for OSPF-4-ADJCHG or OSPF-5-ADJCHG events that show when and why adjacencies changed.

5. Check routing on path devices to see which routes are present or missing.

6. Look up routing history for the affected destination to compare current state against the last known good state.

7. Search ServiceNow for incidents and changes. An OSPF config change or interface change on one device can bring down multiple adjacencies.

## What to look for

- Adjacencies stuck in INIT or EXSTART (MTU mismatch, authentication mismatch)
- Adjacencies that were FULL and dropped (timer expiry, interface flap)
- Interfaces with OSPF enabled on one side but not the other
- Recent changes to OSPF process, network statements, or interface configuration
- Syslog showing repeated adjacency up/down cycles (instability)
