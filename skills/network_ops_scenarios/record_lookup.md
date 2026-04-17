## Scenario: Record Lookup

Use this scenario when the user wants to fetch, show, or get the status/details of an existing ServiceNow record.

## Flow

1. If the request references an `INC...` number, call `get_incident_details(...)`.
2. If the request references a `CHG...` number, call `get_change_request_details(...)`.
3. Return the record details directly.

Do not switch into troubleshooting unless the user is clearly asking to diagnose why something is failing.
