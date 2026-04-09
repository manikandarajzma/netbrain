## Scenario: Connectivity (blocked / denied / port unreachable)

### Step 4 — mandatory calls (all three, in parallel)

Always call these three simultaneously — do not skip any:
1. `ping_device(device=first_hop_device, destination=dest_ip, ...)` — forward ping
2. `ping_device(device=last_hop_device, destination=src_ip, source_interface=last_hop_lan_interface, vrf=dst_vrf)` — reverse ping
3. `trace_reverse_path(source_ip=source_ip, dest_ip=dest_ip)` — reverse path trace

Always write a `## Reverse Path` section in the report based on `trace_reverse_path` output.

### Step 5 — what to do after ping

**Ping FAILED:**
Call `check_routing(devices=[...all path devices...], destination=dest_ip, vrf=src_vrf)` immediately.

**Ping PASSED, port is known:**
Call `test_tcp_port(device=last_hop_device, destination=dest_ip, port=port, vrf=src_vrf)` immediately.
Do not tell the user to run it. Call it yourself.

**Reverse ping FAILED (last_hop_device → src_ip), forward ping passed:**
Call `check_routing(devices=[last_hop_device], destination=src_ip)` to confirm the missing return route.

---

### Root cause patterns

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

**Ping PASSED, TCP connection refused:**
"Layer 3 is healthy. TCP port {port} is actively refused from {last_hop_device}. Either an ACL on {last_hop_device} is blocking port {port}, or the destination is not running a service on that port."
Recommendation: `show ip access-lists` on {last_hop_device}; verify service with `netstat -tlnp | grep {port}` on the destination.

**Ping PASSED, TCP timed out:**
"Layer 3 is healthy. TCP port {port} is silently dropped — a stateful ACL or filter between {last_hop_device} and the destination is discarding the packet."
Recommendation: `show ip access-lists` on {last_hop_device}; check any host-based firewall on the destination.

**Ping PASSED, TCP PASSED:**
"The network path is fully reachable at L3 and L4. The problem is at the application or service layer on the destination."

**Reverse ping FAILED:**
"The return path from {dst_ip} to {src_ip} is broken. {last_hop_device} has no route back to {src_ip}."
Recommendation: On {last_hop_device}: run `show ip route {src_ip}` — if no match, add the missing return route or redistribute the subnet into the routing protocol. If a route exists, run `show ip access-lists` to check for an ACL blocking the return direction.
