## Scenario: Intermittent connectivity (drops in and out / flapping / unstable)

### Key insight

Intermittent issues are almost never routing misconfigurations — they are stability problems. Focus on what is changing over time: OSPF adjacency flaps, interface link events, hardware errors.

### Investigation sequence

**Step 1** — `trace_path(source_ip, dest_ip)` — always first.

**Step 2** — In parallel:
- `search_servicenow(device_names=[...], source_ip=..., dest_ip=...)`
- `get_interface_counters(devices_and_interfaces=[...path_hops...])` — look for CRC/input errors actively incrementing
- `lookup_routing_history(destination_ip=dest_ip)`
- `get_device_syslog(devices=[...path_hops...])` — timestamps of link-down/up and OSPF events are the primary signal

**Step 3** — OSPF stability checks in parallel:
- `check_ospf_neighbors(devices=[...])`
- `check_ospf_interfaces(devices=[...])`
- `lookup_ospf_history(devices=[...])` — look for a trend like "2 → 1 → 2 → 0 → 2"; any variance indicates instability

**Step 4** — Interface detail on suspected devices:
- `get_interface_detail(device=..., interface=...)` — check carrier-transitions counter

**Step 5** — Memory only if live evidence suggests a recurring or still-unresolved pattern:
- `recall_similar_cases(query="...", devices=[...path_hops...])` — use as historical context only, never as current-state proof

---

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
