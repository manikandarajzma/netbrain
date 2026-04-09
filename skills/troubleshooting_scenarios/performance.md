## Scenario: Performance (slow / high latency / degraded throughput)

### Approach

Ping works but traffic is slow or degraded. Focus on interface errors and OSPF instability — these are the most common causes of throughput degradation on a healthy L3 path.

### Step sequence for performance issues

Same as the standard sequence, but pay extra attention to:
- `get_interface_counters` results — actively incrementing input errors, CRC errors, or output drops indicate a physical or hardware problem.
- `lookup_ospf_history` — frequent neighbor count changes indicate OSPF flapping, causing micro-outages and re-convergence delays.
- Ping RTT from `ping_device` — if RTT is high (> 10ms on a LAN), suspect interface errors or QoS misconfiguration.
- `trace_reverse_path` — if the return path is different from the forward path, asymmetric routing may cause TCP performance issues (RSTs, retransmits).

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
