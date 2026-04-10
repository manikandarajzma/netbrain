## Scenario: Connectivity (blocked / denied / port unreachable)

### Investigation sequence

**Step 1** — `trace_path(source_ip, dest_ip)` — always first.

**Step 2** — In parallel:
- `search_servicenow(device_names=[...], source_ip=..., dest_ip=..., port=...)`
- `get_interface_counters(devices_and_interfaces=[...path_hops...])`
- `lookup_routing_history(destination_ip=dest_ip)`

**Step 3** — OSPF checks in parallel on path devices + historically known devices:
- `check_ospf_neighbors(devices=[...])`
- `check_ospf_interfaces(devices=[...])`
- `lookup_ospf_history(devices=[...])`

### Step 4 — mandatory calls (all three, in parallel)

Always call these three simultaneously — do not skip any:
1. `ping_device(device=first_hop_device, destination=dest_ip, ...)` — forward ping
2. `ping_device(device=last_hop_device, destination=src_ip, source_interface=last_hop_lan_interface, vrf=dst_vrf)` — reverse ping
3. `trace_reverse_path(source_ip=source_ip, dest_ip=dest_ip)` — reverse path trace

Always write a `## Reverse Path` section in the report based on `trace_reverse_path` output.

### Step 5 — mandatory tool calls after ping

**Ping FAILED:**
Call `check_routing(devices=[...all path devices...], destination=dest_ip, vrf=src_vrf)` immediately.

**Ping PASSED + port is known — CALL test_tcp_port BEFORE WRITING ANY REPORT:**
You MUST call `test_tcp_port(device=last_hop_device, destination=dest_ip, port=port, vrf=src_vrf)`.
The investigation is not complete until test_tcp_port has returned a result.
Do NOT write Root Cause or Recommendation before this call returns.
Do NOT use any TCP root cause pattern below unless test_tcp_port has been called and returned.

**Reverse ping FAILED (last_hop_device → src_ip), forward ping passed:**
Call `check_routing(devices=[last_hop_device], destination=src_ip)` to confirm the missing return route.

---

### Root cause patterns

> **TCP patterns (refused / timed out / passed) REQUIRE test_tcp_port to have been called first.**
> If you have not called test_tcp_port, go back and call it now. Do not use these patterns otherwise.

**Ping FAILED — management routing fallback (⚠️ Mgmt fallback in path trace):**
"{device} is routing {dst_ip} via the management interface (default route 0.0.0.0/0). Data-plane interfaces are DOWN. Traffic from {src_ip} to {dst_ip} is blackholed."
Recommendation: On {device}: run `show interfaces` to find the down data-plane interface, then `show ip ospf neighbor` to confirm OSPF adjacency is lost. Restore the interface or OSPF config to re-establish the data-plane route.

**Ping FAILED — no route (trace shows ⚠️ no route to {dst_ip}):**
"{device} has no route to {dst_ip}. The routing table has no match for this prefix — OSPF adjacency is missing or the route was withdrawn."
Recommendation: On {device}: run `show ip ospf neighbor` — if 0 neighbors, OSPF has lost adjacency. Run `show ip route {dst_ip}` to confirm no match.

**Ping FAILED — OSPF misconfiguration (ospf_interface_count=0):**
Do not hedge. State definitively: "OSPF process is running on {device} but no interfaces are participating (ospf_interface_count=0). No `network` command or `ip ospf area` is configured on any interface."
- If history shows prior neighbors: "Historically had {N} neighbor(s) — loss of OSPF routes caused management-interface fallback."
Recommendation: re-add `network <subnet> area <id>` or `ip ospf area <id>` on the relevant interfaces. Verify with `show ip ospf neighbor`.

**Ping PASSED, TCP connection refused** (test_tcp_port returned "refused"):
"Layer 3 is healthy. TCP port {port} is actively refused from {last_hop_device}. Either an ACL on {last_hop_device} is blocking port {port}, or the destination is not running a service on that port."
Recommendation: `show ip access-lists` on {last_hop_device}; verify service with `netstat -tlnp | grep {port}` on the destination.

**Ping PASSED, TCP timed out** (test_tcp_port returned "timeout"):
"Layer 3 is healthy. TCP port {port} is silently dropped — a stateful ACL or filter between {last_hop_device} and the destination is discarding the packet."
Recommendation: `show ip access-lists` on {last_hop_device}; check any host-based firewall on the destination.

**Ping PASSED, TCP PASSED** (test_tcp_port returned "success"):
"Layer 3 and Layer 4 are fully reachable. {last_hop_device} successfully connected to {dest_ip}:{port}. The problem is at the application or service layer on the destination."
Recommendation: "Investigate the application on {dest_ip} — check service logs and verify the process is listening on port {port} (e.g. `netstat -tlnp | grep {port}` on the destination host)."

**Reverse ping FAILED:**
"The return path from {dst_ip} to {src_ip} is broken. {last_hop_device} has no route back to {src_ip}."
Recommendation: On {last_hop_device}: run `show ip route {src_ip}` — if no match, add the missing return route or redistribute the subnet into the routing protocol. If a route exists, run `show ip access-lists` to check for an ACL blocking the return direction.

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
## Firewall Policy
## Vendor-Specific Guidance
## Root Cause
## Recommendation
