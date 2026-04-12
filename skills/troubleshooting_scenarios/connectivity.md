## Scenario: Connectivity (blocked / denied / port unreachable)

This file is the authoritative runbook for connectivity investigations. When troubleshooting connectivity, follow this workflow and these conclusion rules exactly.

### Investigation sequence

**Step 1** — `trace_path(source_ip, dest_ip)` — always first.

**Step 2** — In parallel (use ALL devices = path_hops + dst_gateway_device from trace_path output, if known):
- `search_servicenow(device_names=[...all devices...], source_ip=..., dest_ip=..., port=...)`
- `get_interface_counters(devices_and_interfaces=[...path_hops...])` — mandatory whenever trace_path returns any path_hops_for_counters entries
- `get_all_interfaces(device=dst_gateway_device)` — mandatory when dst_gateway_device is visible in forward trace_path output
- `lookup_routing_history(destination_ip=dest_ip)`

Do NOT use `recall_similar_cases(...)` to establish current device state for connectivity investigations.
Past cases may help later with remediation ideas, but they are NOT evidence of current interface state, neighbor state, reachability, or service status.
Never use recalled past cases as support for `## Root Cause`, `## Additional Findings`, or `## Recommendation` unless live tools in this same run independently corroborate them.

**Step 3** — OSPF checks + syslog in parallel on ALL devices (path_hops + dst_gateway_device):
- `check_ospf_neighbors(devices=[...path_hops + dst_gateway_device...])`
- `check_ospf_interfaces(devices=[...path_hops + dst_gateway_device...])`
- `lookup_ospf_history(devices=[...path_hops + dst_gateway_device...])`
- `get_device_syslog(device=...)` on EVERY device in path_hops + dst_gateway_device — one call per device, all in parallel

### Step 4 — mandatory reverse trace first

Call this before any reverse ping:
1. `trace_reverse_path(source_ip=source_ip, dest_ip=dest_ip)` — reverse path trace

Always write a `## Reverse Path` section in the report based on `trace_reverse_path` output.
Do NOT guess the reverse-side device from the forward path.
The reverse ping source device must come from `trace_reverse_path` metadata (`reverse_first_hop_device`), not from `last_hop_device` in the forward trace.

### Step 5 — mandatory pings after reverse trace

After `trace_reverse_path(...)` returns, call both:
1. `ping_device(device=first_hop_device, destination=dest_ip, ...)` — forward ping
2. `ping_device(device=reverse_first_hop_device, destination=src_ip, source_interface=reverse_first_hop_lan_interface, vrf=dst_vrf)` — reverse ping

### Step 6 — mandatory tool calls after ping / reverse trace

**Ping FAILED:**
Call `check_routing(devices=[...all path devices...], destination=dest_ip, vrf=src_vrf)` immediately.

**If forward trace ended early (`no route`, management fallback, or path stopped before reaching destination), use `trace_reverse_path` to identify the destination gateway device before writing Root Cause.**
Treat the last switch/router that still has the destination subnet connected in the reverse trace as `dst_gateway_device`, even if it never appeared in the forward trace.
Once inferred, you MUST run these on that destination gateway device before writing the report:
- `get_all_interfaces(device=dst_gateway_device)`
- `check_ospf_neighbors(devices=[dst_gateway_device])`
- `check_ospf_interfaces(devices=[dst_gateway_device])`
- `lookup_ospf_history(devices=[dst_gateway_device])`
- `get_device_syslog(device=dst_gateway_device)`

When syslog shows an OSPF adjacency teardown with a local interface IP (for example `interface 169.254.0.5 adjacency dropped`), you MUST correlate that IP to the owning interface using `get_all_interfaces(device=dst_gateway_device)` before writing Root Cause.
If the interface owning that local OSPF IP is DOWN, state explicitly that the interface-down event caused the OSPF adjacency loss.
Do not stop at "OSPF adjacency dropped" when you can tie the syslog IP to a specific down interface.
If `get_device_syslog(...)` returns a line such as `Correlated OSPF syslog IP 169.254.0.5 -> Ethernet2 (down ...)`, treat that as explicit evidence and cite that interface/state directly in Root Cause.
Do NOT assume the OSPF adjacency interface is the same interface that holds the destination LAN subnet. A point-to-point OSPF link IP (for example `169.254.x.x`) and a connected destination subnet (for example `10.0.200.0/24`) are usually different interfaces unless a tool explicitly shows otherwise.
If you cannot map the syslog IP to a named interface with tool output, say exactly that: "arista-ai4 lost OSPF adjacency on the peering interface identified by local IP 169.254.0.5." Do NOT rewrite that as "the interface associated with 10.0.200.0/24" or "the interface connected to 10.0.200.0/24" unless a tool explicitly proves it.

If the reverse trace identifies a more likely failing destination gateway than the source-side first hop, prioritize the destination gateway evidence over source-side symptoms.
Do not finalize the report until the `## Interface Errors` section reflects either the `get_interface_counters` result or the exact reason counter polling was unavailable.

If `lookup_routing_history` reports a "Primary upstream clue" or "Primary OSPF peering to troubleshoot" such as "`ai3 EthernetX <-> ai4 EthernetY`", that pair becomes the primary OSPF investigation target.
Before writing Root Cause, you MUST troubleshoot the peering on BOTH devices in that hint, even if the forward trace only named one side:
- `inspect_ospf_peering(device_a=from_device, interface_a=from_interface, device_b=to_device, interface_b=to_interface, ip_a=from_ip_if_known, ip_b=next_hop_ip)` — preferred first call because it bundles interface states, counters, syslog correlation, bilateral peer-IP tests, and returns a `Diagnosis class` plus `Recommended next action`
- `check_ospf_neighbors(devices=[from_device, to_device])`
- `check_ospf_interfaces(devices=[from_device, to_device])`
- `lookup_ospf_history(devices=[from_device, to_device])`
- `get_all_interfaces(device=from_device)` and `get_all_interfaces(device=to_device)`
- `get_interface_detail(device=from_device, interface=from_interface)` and `get_interface_detail(device=to_device, interface=to_interface)`
- `get_device_syslog(device=from_device)` and `get_device_syslog(device=to_device)`
- `get_interface_counters(devices_and_interfaces=[{\"device\": from_device, \"interfaces\": [from_interface]}, {\"device\": to_device, \"interfaces\": [to_interface]}])`
- `ping_device(device=from_device, destination=next_hop_ip, source_interface=from_interface)` and, if the peer IP is known from tools/history, `ping_device(device=to_device, destination=peer_ip, source_interface=to_interface)`

If that bilateral peering evidence exists, base the root cause and recommendation on that specific adjacency rather than on generic destination-subnet wording.
When `inspect_ospf_peering(...)` returns `Diagnosis class:` and `Recommended next action:`, treat those as the primary reasoning anchor. Do not invent a different recommendation unless another tool returned stronger contradictory evidence later in the same run.
If `inspect_ospf_peering(...)` reports `Evidence summary: both ends of the peering are down/admin-down` or that one named side is down/admin-down, treat that as the primary root cause and say so directly.
If `inspect_ospf_peering(...)` returns `Diagnosis class: peering_one_way_reachability_failure` or `Diagnosis class: peering_bidirectional_reachability_failure` while the interfaces are UP, prefer a physical-layer recommendation first:
- inspect/reseat or replace the cable
- inspect/reseat or replace the optic/transceiver
- check interface counters / CRC / port health
- only after that, revisit OSPF timers or policy
If `Diagnosis class: peering_one_way_reachability_failure`, Root Cause MUST explicitly preserve both facts:
- both peering interfaces are currently UP
- one side cannot reach the peer IP
- the OSPF adjacency on that peering is down/lost as a consequence
Preferred phrasing:
`Both peering interfaces are currently up, but {device} {interface} cannot reach peer IP {peer_ip}. This indicates a one-way or intermittent failure on that side of the peering link, and the OSPF adjacency on that peering is down/lost as a consequence.`
Do not compress this into only `cannot reach the peer IP`; keep the `interfaces are up` fact in the final Root Cause.
If the final report only discusses `dst_gateway_device` and does not mention the upstream learner from the routing-history clue, the investigation is incomplete.

**Port is known — CALL test_tcp_port FROM THE CLOSEST DESTINATION-SIDE DEVICE BEFORE WRITING ANY REPORT:**
If `port` is known and you have a destination-adjacent device (`last_hop_device`, `dst_gateway_device`, or the device physically connected to the destination subnet), you MUST call:
`test_tcp_port(device=closest_destination_side_device, destination=dest_ip, port=port, vrf=dst_vrf_or_default)`

This remains REQUIRED even if the source-side ping failed.
Reason: there can be multiple simultaneous blockers, for example:
- the network path is broken
- and the destination service is also not listening on the requested TCP port

The investigation is not complete until test_tcp_port has returned a result whenever a destination-adjacent device is known.
Do NOT write the final report before this call returns.
If Layer 3 is broken and TCP testing from the destination side shows the service is also unavailable, report the Layer 3 issue as the primary root cause and the TCP/service result as an additional finding in `## Connectivity Test`.
Do NOT suppress the TCP finding just because routing is already broken.

**Reverse ping FAILED (reverse_first_hop_device → src_ip), forward ping passed:**
Call `check_routing(devices=[reverse_first_hop_device], destination=src_ip)` to confirm the missing return route.

### Step 7 — mandatory holistic snapshot before Root Cause / Recommendation

Before writing the final report, call:

`collect_connectivity_snapshot(source_ip=src_ip, dest_ip=dest_ip, port=port_if_known)`

Treat that snapshot as the primary evidence bundle for final reasoning.

The snapshot is discovery-first. It is expected to:
- discover which routing protocols are actually present on each device instead of assuming OSPF
- discover L2 control-plane mode (for example MSTP / RSTP) when relevant instead of assuming generic STP
- summarize topology, route status, relevant interfaces, counters, syslog, protocol state, and service checks in one place
- surface more than one independent blocker when they exist

Do not collapse the investigation into a single story too early.
If the snapshot surfaces multiple independent problems, write:
- `## Root Cause` for the primary blocker that best explains the end-to-end failure
- and preserve the other blockers under `## Additional Findings` or `## Connectivity Test`

When live evidence and historical evidence disagree, prefer live evidence.
If a device or interface could not be queried live in this run, say that explicitly and avoid asserting its current state.

`test_tcp_port(...)` is only one possible service-layer check. Treat it as part of the holistic snapshot when a port is known, not as the only additional blocker worth reporting.

---

### Root cause patterns

> **TCP patterns (refused / timed out / passed) REQUIRE test_tcp_port to have been called first whenever a destination-side device is available.**
> If you have not called test_tcp_port, go back and call it now. Do not use these patterns otherwise.

**Ping FAILED — management routing fallback (⚠️ Mgmt fallback in path trace):**
"{device} is routing {dst_ip} via the management interface (default route 0.0.0.0/0). Data-plane interfaces are DOWN. Traffic from {src_ip} to {dst_ip} is blackholed."
Recommendation: Restore the down data-plane interface and OSPF on {device}.

**Ping FAILED — no route (trace shows ⚠️ no route to {dst_ip}):**
The device with no route is a symptom. The root cause is on dst_gateway_device (identified in trace_path output).
Use get_all_interfaces, check_ospf_neighbors, and lookup_routing_history on dst_gateway_device to find why it stopped advertising the route.
- If egress interface is DOWN: "{interface} on {dst_gateway_device} is down. OSPF adjacency was lost and {dest_subnet} was withdrawn, leaving {no_route_device} with no route to {dest_ip}."
  Recommendation: Bring {interface} on {dst_gateway_device} back up. OSPF will reconverge. Do NOT recommend static routes.
- If interface UP but OSPF neighbor missing: "{dst_gateway_device} lost OSPF adjacency via {interface}. {dest_subnet} withdrawn."
  Recommendation: Check the peer link and neighbor on the OSPF-facing interface {interface} on {dst_gateway_device} — timers, area, authentication, and whether the far-end interface is up.

When forward and reverse evidence disagree, prefer the device closest to the failed destination subnet:
- If reverse trace reaches `{dst_gateway_device}` and that device has a DOWN interface or lost OSPF adjacency, that is the root cause.
- If syslog names a local OSPF interface IP and `get_all_interfaces` shows the owning interface is down, say that the interface-down condition is the direct cause and the OSPF loss is the consequence.
- If syslog names a local OSPF interface IP, `get_all_interfaces` shows that interface is UP, and interface counters are clean, do NOT invent a physical issue. State that the adjacency failed despite the local interface being up/clean, so the next most likely causes are neighbor-side failure or OSPF parameter mismatch on that specific peering link.
- If routing history identifies a specific historical peering pair (for example `ai3 EthernetX <-> ai4 EthernetY`), name that pair explicitly in Root Cause and Recommendation. Do not collapse it into a generic statement about the destination subnet interface.
- When such a historical peering pair exists, preferred phrasing is: "`ai3` historically learned `10.0.200.0/24` from `ai4` over the `ai3 EthernetX <-> ai4 EthernetY` OSPF peering. That peering is now down/lost." Avoid summarizing this as a problem on "the 10.0.200.0/24 interface."
- Do NOT treat "`inactivity timer expired`" as the root cause by itself. That message only proves the adjacency died. The root cause must come from the bilateral peering evidence: local interface down, peer interface down, failed ping across the peering IPs, missing OSPF on one side, or OSPF parameter mismatch/authentication issue.
- Do NOT blame `{no_route_device}` just because it reports "no route". That device is only reporting the missing route.
- Do NOT pick a source-side `ospf_interface_count=0` root cause if a destination-gateway interface-down / adjacency-loss condition explains the withdrawal more directly.
- Forbidden phrasing unless explicitly proven by tools: "the interface associated with the 10.0.200.0/24 subnet", "the interface connected to the 10.0.200.0/24 subnet", or any wording that equates the OSPF peering IP/interface with the destination LAN interface.

**Ping FAILED — OSPF misconfiguration (ospf_interface_count=0):**
Use this ONLY when other evidence supports misconfiguration. `ospf_interface_count=0` alone is not enough, because a down interface can also produce zero currently reported OSPF interfaces.
Only state misconfiguration definitively if ALL of the following are true:
- `ospf_interface_count=0`
- `get_all_interfaces` does NOT show the relevant OSPF-facing interface down
- syslog does NOT show a link-down or adjacency-loss event that explains the failure
- OSPF history does NOT indicate the device recently had neighbors that were lost
- routing/history evidence does not point to a downstream destination-gateway withdrawal

If those conditions are met, state definitively: "OSPF process is running on {device} but no interfaces are participating (ospf_interface_count=0). No `network` command or `ip ospf area` is configured on any interface."
- If history shows prior neighbors: "Historically had {N} neighbor(s) — loss of OSPF routes caused management-interface fallback."
Recommendation: re-add `network <subnet> area <id>` or `ip ospf area <id>` on the relevant interfaces.

Only use this OSPF misconfiguration pattern as the primary root cause when no downstream destination-gateway failure better explains the outage.
If another device physically attached to the destination subnet has a down interface or lost adjacency, that downstream failure takes precedence.
Never claim the same OSPF misconfiguration simultaneously on every device in the path unless every device has independent corroborating evidence and there is no more specific failing interface/device to blame.

**Ping PASSED, TCP connection refused** (test_tcp_port returned "refused"):
"Layer 3 is healthy. TCP port {port} is actively refused from {last_hop_device}. Either an ACL on {last_hop_device} is blocking port {port}, or the destination is not running a service on that port."

**Ping PASSED, TCP timed out** (test_tcp_port returned "timeout"):
"Layer 3 is healthy. TCP port {port} is silently dropped — a stateful ACL or filter between {last_hop_device} and the destination is discarding the packet."

**Ping PASSED, TCP PASSED** (test_tcp_port returned "success"):
"Layer 3 and Layer 4 are fully reachable. The problem is at the application or service layer on the destination."

**Reverse ping FAILED:**
"The return path from {dst_ip} to {src_ip} is broken. {reverse_first_hop_device} has no route back to {src_ip}."

---

### Report format

Use these exact headers (omit any with nothing to report):

## Path Summary
## Reverse Path
## ServiceNow
## Interface Errors
## OSPF Analysis
## Routing Analysis
## Connectivity Test
## Additional Findings
## Root Cause
## Recommendation
