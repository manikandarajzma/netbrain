## Scenario: Intermittent connectivity (drops in and out / flapping / unstable)

### Key insight

Intermittent issues are almost never routing misconfigurations — they are stability problems. Focus on what is changing over time: OSPF adjacency flaps, interface link events, hardware errors.

### Step sequence for intermittent issues

Follow the standard sequence. Pay close attention to:
- `lookup_ospf_history` — look for a trend like "2 → 1 → 2 → 0 → 2". Any variance in neighbor count indicates instability.
- `get_device_syslog` — look for repeated link-down/up events, OSPF adjacency drops, or err-disable events. The timestamp of these events correlates the disruption.
- `get_interface_counters` — CRC errors or input errors that are actively incrementing point to a flapping physical link.
- `get_interface_detail` — check for frequent carrier transitions (input resets, carrier transitions counter).

### Root cause patterns

**OSPF adjacency flapping:**
"OSPF neighbor count on {device} has been unstable: {history trend}. Each drop causes a brief routing black hole until reconvergence. Syslog confirms adjacency events at {timestamps}."
Recommendation:
- Check OSPF hello/dead timer mismatch with the peer (`show ip ospf neighbor detail`)
- Check for BFD misconfiguration if BFD is used
- Check physical stability of the interface OSPF runs on (`show interfaces {interface}`)

**Physical link instability:**
"Syslog on {device} shows repeated link-down/up events on {interface} at {timestamps}. This is causing periodic traffic drops while OSPF reconverges."
Recommendation:
- Inspect physical cable and SFP on {interface} and the peer port
- Check for carrier-transitions counter (`show interfaces {interface}`)
- Consider replacing the SFP or cable if errors persist

**Err-disable cycling:**
"Syslog shows {interface} repeatedly entering err-disable state. The port is flapping due to an err-disable trigger (BPDU guard, storm control, or similar)."
Recommendation:
- Identify the err-disable reason: `show interfaces {interface} status err-disabled`
- Fix the root trigger before re-enabling the port

**No evidence of flapping (path and interfaces stable):**
If OSPF history is flat, interfaces are clean, and syslog shows no events — the problem may be upstream of the first-hop device or at the application layer.
Root cause: "No network-layer instability detected. OSPF is stable and interfaces are clean. The intermittent issue may be at the application layer or in a segment not visible to this inventory."
