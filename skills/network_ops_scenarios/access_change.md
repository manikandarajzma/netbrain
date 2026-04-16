## Scenario: Access / Rule / Port Change Request

Use this scenario when the user is asking to:
- open a port
- allow or deny traffic
- whitelist access
- create or update a firewall or security rule
- document a network access change

## Workflow

1. If source and destination IPs are present, call `trace_path(source_ip, dest_ip)` to identify related devices.
2. Call `search_servicenow(...)` when recent incidents or approved changes may matter.
3. Produce a structured access/network change request.

Do not claim to inspect live firewall policy state directly.

## Output format

```
## Network Change Request

| Field            | Value                        |
|------------------|------------------------------|
| Source IP        | <source_ip>                  |
| Destination IP   | <dest_ip>                    |
| Destination Port | <port> / <protocol>          |
| Related Devices  | <hostnames from path>        |
| Requested Action | allow / deny / modify rule   |

## Justification
<one sentence describing the business need>

## Existing Coverage
<related incidents or approved changes, or "None found">

## Recommended Rule
Source: <source_ip or subnet>
Destination: <dest_ip or subnet>
Port/Protocol: <port>/<protocol>
Action: <allow or deny>
Affected devices: <devices from path or CI>
```
