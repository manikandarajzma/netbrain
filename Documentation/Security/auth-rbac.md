# Atlas Authentication & Group-Based Access

## Overview

Atlas uses Microsoft Entra ID (OIDC) for authentication. Access is controlled by group membership — no roles, no PIM. When a user signs in, their Entra ID group memberships are read from the OIDC id_token and mapped to an access level.

**No local passwords.** All backend credentials (NetBrain, Panorama, Splunk, etc.) are stored in Azure Key Vault.

---

## How Sign-In Works

1. User clicks **Sign in with Microsoft** and authenticates with Entra ID
2. Microsoft returns an id_token containing a `groups` claim with the user's group Object IDs
3. Atlas checks those Object IDs against the configured group IDs (`ATLAS_ADMIN_GROUP_ID`, `ATLAS_NETADMIN_GROUP_ID`)
4. If a match is found, the access level is stored in a signed session cookie
5. If no match is found, the user is rejected with: _"Your account is not a member of an authorised group."_

---

## Access Levels

Two access levels are defined in `auth.py`, plus an internal `guest` fallback for invalid/expired sessions:

| Access Level | Tools Allowed                                                              | Sidebar Categories  |
|--------------|----------------------------------------------------------------------------|---------------------|
| Full access  | All tools                                                                  | All categories      |
| Limited      | Network path queries, Panorama object/group queries                        | Atlas, Panorama     |
| Guest        | None                                                                       | None                |

Each Entra ID group maps to one of the two active access levels via the env vars below.

---

## How Tool Access is Enforced

Group membership is checked **twice**:

1. **At sign-in** — `extract_group_from_token()` in [auth.py](../../auth.py) reads the `groups` claim from the id_token and resolves it to an access level, or rejects the login.

2. **On every tool call** — `_check_tool_access()` in [chat_service.py](../../chat_service.py) reads the access level from the session cookie and checks it against `GROUP_ALLOWED_TOOLS` before running any tool. This check is code-enforced and cannot be bypassed by prompt injection.

---

## Environment Variables

| Variable                  | Required | Description                                                              |
|---------------------------|----------|--------------------------------------------------------------------------|
| `AUTH_MODE`               | Yes      | Must be `oidc`                                                           |
| `AZURE_CLIENT_ID`         | Yes      | App Registration client ID                                               |
| `AZURE_CLIENT_SECRET`     | Yes      | App Registration client secret                                           |
| `AZURE_TENANT_ID`         | Yes      | Azure tenant ID                                                          |
| `ATLAS_ADMIN_GROUP_ID`    | Yes      | Object ID of the full-access group                                       |
| `ATLAS_NETADMIN_GROUP_ID` | Yes      | Object ID of the limited-access group                                    |
| `SESSION_SECRET`          | No       | Fixed secret for signing session cookies — set for multi-instance deployments so sessions survive restarts |
| `OAUTH_STATE_SECRET`      | No       | Fixed secret for OAuth state cookie — set for multi-instance deployments |
