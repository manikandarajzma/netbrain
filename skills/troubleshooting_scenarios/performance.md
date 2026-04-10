## Scenario: Performance (slow / high latency / degraded throughput)

### Key insight

Ping works but traffic is slow or degraded. Focus on interface errors and OSPF instability — these are the most common causes of throughput degradation on a healthy L3 path.

### Investigation sequence

**Step 1** — `trace_path(source_ip, dest_ip)` — always first.

**Step 2** — In parallel:
- `search_servicenow(device_names=[...], source_ip=..., dest_ip=...)`
- `get_interface_counters(devices_and_interfaces=[...path_hops...])` — actively incrementing errors are the primary signal
- `lookup_routing_history(destination_ip=dest_ip)`
- `get_device_syslog(devices=[...path_hops...])` — look for recurring error events

**Step 3** — OSPF checks in parallel on path devices + historically known devices:
- `check_ospf_neighbors(devices=[...])`
- `check_ospf_interfaces(devices=[...])`
- `lookup_ospf_history(devices=[...])` — look for neighbor count changes (flapping)

**Step 4** — Ping to measure RTT:
- `ping_device(device=first_hop_device, destination=dest_ip, source_interface=first_hop_lan_interface, vrf=src_vrf)`
- `trace_reverse_path(source_ip=source_ip, dest_ip=dest_ip)` — asymmetric path causes TCP issues

---

### Root cause patterns

**High interface error counters:**
"Input errors / CRC errors incrementing on {interface} at {device} indicate a physical layer problem (bad cable, SFP, or duplex mismatch). This causes packet retransmissions and throughput degradation."
Recommendation: Check cable and SFP on {interface} and its peer port. Check for duplex mismatch (`show interfaces {interface}`).

**OSPF flapping (history shows neighbor count changing):**
"OSPF neighbor count on {device} has fluctuated ({history trend}). Each reconvergence event causes a brief traffic black hole and TCP retransmission storm."
Recommendation: Check OSPF hello/dead timers, interface stability (`show ip ospf neighbor detail`), and syslog for adjacency events.

**Asymmetric routing:**
"`trace_reverse_path` shows traffic returning via a different path than the forward path. This can cause TCP RSTs if a stateful device sees only one direction of a flow."
Recommendation: Investigate routing policy on the return-path devices. Ensure symmetric routing or configure stateful devices to allow asymmetric flows.

**Output drops / queue drops:**
"Output drops on {interface} at {device} indicate congestion. Traffic is being queued and dropped before transmission."
Recommendation: Review QoS policy, interface bandwidth, and traffic rates. Consider traffic shaping or upgrading the link.

**Clean counters, no OSPF issues:**
If all interfaces are clean and OSPF is stable, the bottleneck is likely at the application or server layer.
Root cause: "Network path is healthy (no interface errors, OSPF stable, RTT normal). Performance issue is at the application or server layer."

---

### Report format

Use these exact headers (omit any with nothing to report):

## Path Summary
## ServiceNow
## Interface Errors
## OSPF Analysis
## Vendor-Specific Guidance
## Root Cause
## Recommendation
