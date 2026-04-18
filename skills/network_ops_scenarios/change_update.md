## Scenario: Change Update

Use this scenario when the user wants to update, close, reassign, or add work notes to an existing change request.

## Flow

1. Gather the needed update fields from the user:
   - `number`
   - plus any of: `state`, `work_notes`, `assigned_to`, `close_notes`
2. Call `update_servicenow_change_request(...)`.
3. Report that the update is staged for approval and Atlas is waiting for user confirmation.

Do not turn this into a generic change-creation flow.
