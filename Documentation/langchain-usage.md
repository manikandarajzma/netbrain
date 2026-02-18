# LangChain — Functionality and Usage

This document explains what LangChain is, which of its components are used in this codebase, and exactly how each is used, with code snippets from the actual source files.

---

## 1. What LangChain Is

LangChain is a Python framework for building applications that call LLMs. Rather than calling an LLM's API directly, LangChain provides:

- **Model wrappers** — a uniform interface to many LLM providers (Ollama, OpenAI, Anthropic, etc.) so switching models requires changing one line
- **Prompt templates** — reusable, parameterised message builders
- **Output parsers / structured outputs** — tools to constrain LLM responses to typed Python objects instead of raw text
- **Chains (LCEL)** — the `|` pipe operator that composes `prompt | model | parser` into a single callable

This codebase uses LangChain purely as a thin wrapper over a **local Ollama** instance running `qwen2.5:14b`. No cloud LLM APIs are called.

### Packages imported

| Package | What it provides | Used in |
|---|---|---|
| `langchain_ollama` | `ChatOllama` — Ollama LLM wrapper | `tools/shared.py`, `mcp_client_tool_selection.py`, `chat_service.py` |
| `langchain_core.prompts` | `ChatPromptTemplate` — message template builder | `tools/shared.py`, `tools/panorama_tools.py` |

---

## 2. LangChain Components Used

### 2.1 `ChatOllama` — The LLM Wrapper

`ChatOllama` is a LangChain model class that sends chat-style messages to a locally running Ollama server over HTTP and returns a `AIMessage` response object.

```python
# tools/shared.py (lines 74-92)
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen2.5:14b",       # model name served by Ollama
    temperature=0.0,           # 0 = deterministic, no sampling randomness
    base_url="http://localhost:11434",  # Ollama HTTP server
)
```

It exposes two invocation methods:

| Method | Type | Used for |
|---|---|---|
| `llm.invoke(prompt_text)` | Synchronous | Tool selection (fallback JSON path) |
| `await llm.ainvoke(messages)` | Async | Scope check, AI enrichment, final answer synthesis |

The return value is always an `AIMessage` object. The text content is accessed via `response.content`.

```python
response = llm.invoke("some prompt")
text = response.content   # str
```

---

### 2.2 `ChatPromptTemplate` — Prompt Builder

`ChatPromptTemplate` builds a list of typed chat messages (`system`, `human`, `assistant`) from a template string. The template uses `{variable}` placeholders that are filled in at call time.

Two factory methods are used:

**`from_template(template)`** — single human message:

```python
# Used in tools/netbox_tools.py (pattern)
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    "Summarise the following rack configuration:\n{data}"
)
# Equivalent to: [HumanMessage(content="Summarise the following rack configuration:\n<data>")]
```

**`from_messages([(role, text), ...])`** — multi-turn message list (system + human):

```python
# tools/panorama_tools.py (lines 1034-1041)
from langchain_core.prompts import ChatPromptTemplate

analysis_prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),                              # sets model behaviour
    ("human", "Analyze this Panorama query result:\n{query_result}")  # user turn
])

formatted_messages = analysis_prompt_template.format_messages(
    query_result=json.dumps(result, indent=2)   # fills {query_result} placeholder
)
# formatted_messages is now a list: [SystemMessage(...), HumanMessage(...)]
```

The formatted message list is then passed directly to `llm.ainvoke()`.

---

### 2.3 `llm.with_structured_output(Schema)` — Pydantic Structured Outputs

This is the most important LangChain feature used in the codebase. Instead of asking the LLM to produce JSON as free text (which it sometimes formats incorrectly), `with_structured_output` constrains the LLM to populate a **Pydantic model** directly.

```python
# mcp_client_tool_selection.py (lines 262-268)
from pydantic import BaseModel, Field

class ToolSelection(BaseModel):
    entity_analysis: Optional[str]
    tool_name: Optional[str]
    needs_clarification: bool
    clarification_question: Optional[str]
    parameters: ToolParameters   # nested Pydantic model

structured_llm = llm.with_structured_output(ToolSelection)
response: ToolSelection = structured_llm.invoke(prompt_text)
# response is a fully typed Python object — no JSON parsing needed
```

The benefit over raw JSON: the LLM cannot accidentally wrap output in markdown fences, use Python `None`/`True`/`False` instead of JSON `null`/`true`/`false`, or produce malformed JSON. LangChain handles the schema enforcement internally.

---

### 2.4 LCEL — The Pipe `|` Operator

LangChain Expression Language (LCEL) lets you compose a prompt template and a model into a chain using `|`. Calling `.invoke()` on the chain runs both in sequence:

```python
# Pattern used in tools/netbox_tools.py
chain = ChatPromptTemplate.from_template("Summarise:\n{data}") | llm

result = chain.invoke({"data": raw_data})
# Equivalent to:
#   messages = prompt.format_messages(data=raw_data)
#   result   = llm.invoke(messages)
```

In this codebase LCEL chains are used only for AI enrichment inside MCP tool handlers. Tool selection and scope checks call `llm.invoke()` / `llm.ainvoke()` directly because they need finer control (structured output wrappers, async wait_for timeouts).

---

## 3. Where LangChain Is Used

### 3.1 LLM Singleton — `tools/shared.py`

The single `ChatOllama` instance is created lazily and stored on the shared `mcp` object so all domain tool modules reuse it without creating duplicates.

```python
# tools/shared.py (lines 74-111)
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate  # re-exported for convenience

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")

def _get_llm():
    """Lazy singleton — only initialised on first call."""
    if mcp.llm is None:
        try:
            mcp.llm = ChatOllama(
                model=OLLAMA_MODEL,
                temperature=0.0,
                base_url=OLLAMA_BASE_URL,
            )
        except Exception as e:
            mcp.llm = False            # False = tried and failed, don't retry
            mcp.llm_error = {"error": str(e)}
    return mcp.llm if mcp.llm is not False else None
```

`ChatPromptTemplate` is imported here and re-exported so domain tool modules can write `from tools.shared import ChatPromptTemplate` instead of importing from `langchain_core` directly.

---

### 3.2 Scope Classification — `chat_service.py`

Before tool selection, the LLM is used as a binary classifier to decide if the user's query is within the system's scope. A short prompt with a 5-second timeout asks for `IN_SCOPE` or `OUT_OF_SCOPE`.

```python
# chat_service.py (lines 349-392)
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "qwen2.5:14b"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    temperature=0.0,
)

scope_check_prompt = f"""You are a scope classifier for a network infrastructure assistant.
...
Query: "{prompt}"
RESPOND WITH ONLY "IN_SCOPE" OR "OUT_OF_SCOPE".
"""

response = await asyncio.wait_for(
    llm.ainvoke(scope_check_prompt),   # ainvoke = async, awaitable
    timeout=5.0                         # short timeout — fail fast
)

result_text = response.content.strip().upper()
# "OUT_OF_SCOPE" → block query; anything else → allow through
```

Note: this LangChain call creates a fresh `ChatOllama` instance per call (not the shared singleton), because the scope check is done on the client side (`chat_service.py`) while the singleton lives on the server side (`tools/shared.py`).

---

### 3.3 Tool Selection — `mcp_client_tool_selection.py`

This is the primary LangChain usage. `ChatOllama` + `with_structured_output` converts natural language into a typed `ToolSelection` object containing the tool name and extracted parameters.

```python
# mcp_client_tool_selection.py (lines 228-294)
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

# --- Pydantic schema the LLM must populate ---
class ToolParameters(BaseModel):
    ip_address:   Optional[str]   # e.g. "10.0.0.1"
    device_name:  Optional[str]   # e.g. "leander-dc-leaf1"
    rack_name:    Optional[str]   # e.g. "A4"
    source:       Optional[str]   # source IP for path queries
    destination:  Optional[str]   # destination IP for path queries
    protocol:     Optional[str]
    port:         Optional[str]
    limit:        Optional[int]   # for "latest N" Splunk queries
    site_name:    Optional[str]
    # ... more fields

class ToolSelection(BaseModel):
    entity_analysis:       Optional[str]   # LLM's reasoning (logged, not returned to user)
    tool_name:             Optional[str]   # exact registered tool name
    needs_clarification:   bool
    clarification_question: Optional[str]
    parameters:            ToolParameters

# --- Invocation ---
llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.0)

structured_llm = llm.with_structured_output(ToolSelection)

prompt_text = build_tool_selection_prompt(user_query, tools_description, conversation_history)
response: ToolSelection = structured_llm.invoke(prompt_text)

# response is a typed Python object — access fields directly
tool_name   = response.tool_name          # "query_network_path"
params      = response.parameters.source  # "10.0.0.1"
needs_clarification = response.needs_clarification  # False
```

**Fallback path** — if `with_structured_output` fails (e.g. the model doesn't support it), the code falls back to calling `llm.invoke()` directly and parsing the raw text response as JSON:

```python
# mcp_client_tool_selection.py (lines 300-330)
response = llm.invoke(prompt_text)           # returns AIMessage
content  = response.content                  # raw string

# Find first { ... last } and fix Python literals → valid JSON
first_brace = content.find('{')
last_brace  = content.rfind('}')
json_str    = content[first_brace:last_brace + 1]
json_str    = re.sub(r'\bNone\b',  'null',  json_str)
json_str    = re.sub(r'\bTrue\b',  'true',  json_str)
json_str    = re.sub(r'\bFalse\b', 'false', json_str)
parsed      = json.loads(json_str)
```

**Final answer synthesis** — also in this file, `ainvoke` is used asynchronously when generating a human-readable error message after all retries fail:

```python
# mcp_client_tool_selection.py (lines 393-408)
llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.3)
# temperature=0.3 here (not 0.0) — slight variation is fine for error messages

response = await asyncio.wait_for(
    llm.ainvoke(prompt_text),
    timeout=15.0
)
return response.content.strip()
```

---

### 3.4 Panorama Tool — AI Enrichment of `query_panorama_ip_object_group`

After querying the Panorama API for which address group an IP belongs to, the raw result dict is handed to the LLM to produce a 2–4 sentence plain-English summary. The system prompt explicitly forbids markdown tables to avoid formatting clutter in the UI.

```python
# tools/panorama_tools.py (lines 1017-1080)
llm = _get_llm()   # shared singleton from tools/shared.py
if llm is not None and "error" not in result:
    from langchain_core.prompts import ChatPromptTemplate

    system_prompt = """You are a network security assistant. The UI will display the data in proper tables.
Your job is to write a SHORT narrative summary only (2-4 sentences).
Do NOT use markdown tables, pipe characters, or column layouts.
STARTS with the direct answer: which address group(s) the IP is in."""

    analysis_prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Analyze this Panorama query result:\n{query_result}")
    ])

    formatted_messages = analysis_prompt_template.format_messages(
        query_result=json.dumps(result, indent=2)   # serialised API response
    )

    response = await asyncio.wait_for(
        llm.ainvoke(formatted_messages),   # async, takes list of messages
        timeout=30.0
    )
    content = response.content

    result["ai_analysis"] = {"summary": content}
    # result now has both raw API data AND the LLM-written summary
```

If the LLM call fails, a static fallback summary is built from the raw counts (no LangChain involved):

```python
except Exception:
    result["ai_analysis"] = {
        "summary": f"IP {ip_address} found in {addr_objects_count} address object(s) and {addr_groups_count} address group(s)."
    }
```

---

### 3.5 Panorama Tool — AI Enrichment of `query_panorama_address_group_members`

The second Panorama tool uses the same pattern but the system prompt requests **markdown tables** (two of them: one for address objects, one for policies), because this result set is larger and tabular layout is more readable.

```python
# tools/panorama_tools.py (lines 1545-1607)
system_prompt = """You are a network security assistant. Provide a concise summary in TABLE FORMAT.

**Table 1: Object group details**
Columns: Address Object Name, Type, IP Address/Value, Location

**Table 2: Policy details**
Columns: Address Group, Policy Name, Policy Type, Rulebase, Action/NAT Type, Source, Destination, Services, Location
"""

analysis_prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Analyze this Panorama address group members query result:\n{query_result}")
])

formatted_messages = analysis_prompt_template.format_messages(
    query_result=json.dumps(result, indent=2)
)

response = await asyncio.wait_for(
    llm.ainvoke(formatted_messages),
    timeout=30.0
)

result["ai_analysis"] = {"summary": response.content}
```

---

## 4. LangChain Call Summary

| Location | LangChain API used | Purpose | `temperature` | Timeout |
|---|---|---|---|---|
| `tools/shared.py` | `ChatOllama(...)` | Creates shared LLM singleton | `0.0` | n/a |
| `chat_service.py` | `llm.ainvoke(str)` | Binary scope classifier | `0.0` | 5 s |
| `mcp_client_tool_selection.py` | `llm.with_structured_output(ToolSelection).invoke(str)` | Tool name + parameter extraction | `0.0` | none (sync) |
| `mcp_client_tool_selection.py` | `llm.invoke(str)` | Fallback raw JSON extraction | `0.0` | none (sync) |
| `mcp_client_tool_selection.py` | `llm.ainvoke(str)` | Final answer synthesis on failure | `0.3` | 15 s |
| `tools/panorama_tools.py` | `ChatPromptTemplate.from_messages` + `llm.ainvoke(list)` | Narrative summary of IP object group result | `0.0` (inherited) | 30 s |
| `tools/panorama_tools.py` | `ChatPromptTemplate.from_messages` + `llm.ainvoke(list)` | Markdown table summary of address group members | `0.0` (inherited) | 30 s |

---

## 5. Why LangChain Over Direct HTTP Calls

Calling Ollama's API directly (`POST /api/chat`) is straightforward, so why use LangChain?

1. **`with_structured_output`** — this one feature alone justifies the dependency. Getting a reliably typed `ToolSelection` object from the LLM without any JSON parsing fragility is not trivial to implement from scratch.

2. **Provider portability** — swapping from Ollama to OpenAI or another provider requires only changing the import and constructor:
   ```python
   # Ollama (current)
   from langchain_ollama import ChatOllama
   llm = ChatOllama(model="qwen2.5:14b", base_url="http://localhost:11434")

   # OpenAI (hypothetical swap)
   from langchain_openai import ChatOpenAI
   llm = ChatOpenAI(model="gpt-4o")
   ```
   All downstream `.invoke()`, `.ainvoke()`, `with_structured_output()` calls remain unchanged.

3. **`ChatPromptTemplate`** — separates prompt structure from prompt content, making system prompts and human turns independently editable without string concatenation.

4. **Async support** — `ainvoke` integrates cleanly with the `asyncio`/FastAPI event loop already used throughout the codebase, with no extra threading needed.
