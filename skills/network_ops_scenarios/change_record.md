## Scenario: Generic ServiceNow Change Request

Use this scenario when the user is asking to create/open/submit a generic ServiceNow change request and is NOT primarily asking for an access rule, firewall policy, port opening, whitelist, or network rule document.

## Required fields

Before creating the change request, make sure these are present:
1. `short_description`
2. `description`
3. `ci_name`
4. `justification`
5. `implementation_plan`

If any are missing, ask for the missing fields directly.

If the user provides those fields in any structured form — numbered list, bullets, inline labels, or `Field: value` pairs — treat them as the answers and proceed.

If `ci_name` is already present, do not call `trace_path(...)` just to identify the same CI again.

Do NOT ask for:
- source IP
- destination IP
- destination port
- recommended firewall rule

unless the user is explicitly asking for an access/rule change rather than a generic change record.

## Flow

1. Optionally call `search_servicenow(...)` for the referenced CI if it adds useful context.
2. Call `create_servicenow_change_request(...)`.
3. Report that the change request is staged for approval and Atlas is waiting for user confirmation.

## Output format

Use a direct ServiceNow record confirmation format, not a firewall-rule template.

Preferred structure:

```
Change Request Proposed

Short Description: <short_description>
Configuration Item: <ci_name>
Justification: <justification>
Implementation Plan: <implementation_plan>

Related ServiceNow context:
<brief summary or "None found">

Next step: user confirms before Atlas executes the write
```

Do NOT output:
- `## Network Change Request`
- `Source IP`
- `Destination IP`
- `Destination Port`
- `Recommended Rule`

unless the request is specifically an access/rule/policy change.
