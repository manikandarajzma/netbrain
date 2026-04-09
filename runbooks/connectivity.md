---
name: Connectivity Troubleshooting
triggers: cannot reach, unreachable, connectivity, ping, port, blocked, not working, connection refused, timeout, 443, 80, 22, ssh, https, http
---

Use this runbook when a source IP cannot reach a destination IP, optionally on a specific port.

## What to investigate

1. Trace the forward path (src → dst) and the return path (dst → src). Asymmetric routing is a common cause of connectivity failures — always check both.

2. Search ServiceNow for incidents and changes on all path devices. Recent changes frequently explain connectivity failures.

3. Check interface counters on every interface in both paths. Actively incrementing CRC errors or drops indicate a physical or congestion problem.

4. Ping the destination from the last-hop device (the one directly connected to the destination subnet). This is the most useful test — if it fails, the destination host is unreachable at Layer 3. If it passes, the problem is Layer 4 or above.

5. If a specific port was given and ping passes, run a TCP port test from the last-hop device. This confirms whether the service is accepting connections at the network level.

6. If ping fails, check routing on all path devices and look up routing history to find what changed.

7. Pull syslog from any device showing errors or where the path looks wrong. Look for interface flaps and OSPF adjacency changes.

## What to look for

- Asymmetric paths (forward and return using different devices/interfaces)
- Missing or changed routes (compare routing history)
- Actively incrementing error counters (CRC, drops, input errors)
- Interface flaps in syslog around the time the issue started
- Open incidents or recent changes on path devices
- TCP port test failing while ping passes (firewall or application issue)
