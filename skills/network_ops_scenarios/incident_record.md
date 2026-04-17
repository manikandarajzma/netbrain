## Scenario: Incident Record

Use this scenario when the user is asking to create, open, or raise a new ServiceNow incident or ticket.

## Required fields

Before creating the incident, make sure these are present:
1. `short_description`
2. `description`

Optional but useful:
- `ci_name`
- `urgency`
- `impact`

If the user provides those fields in numbered lists, bullets, inline labels, or `Field: value` pairs, use them directly.

If `ci_name` is not given but source and destination IPs are present, you may call `trace_path(...)` first to identify the most relevant device.

## Flow

1. If helpful and clearly relevant, call `search_servicenow(...)` for recent related incidents or changes.
2. Call `create_servicenow_incident(...)`.
3. Report the created incident directly.

## Output format

Use a direct incident confirmation format.

Preferred structure:

```
Incident Created

Number: <created incident number>
Short Description: <short_description>
Configuration Item: <ci_name or "Not specified">

Related ServiceNow context:
<brief summary or "None found">
```
