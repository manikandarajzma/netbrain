"""
Documentation search tool — two-step LLM routing.

How it works:
  1. A registry maps each documentation file to a one-line description.
  2. LLM call 1: shown the registry, returns the single most relevant file path.
  3. LLM call 2: shown just the section headings of that file, returns which
     sections are needed to answer the query. Only those sections are returned.

Why two steps?
  File selection: the LLM picks from ~15 options using intent, not keywords.
  Section selection: avoids returning 200-line files when the user asked about
  one specific thing (e.g. ".env variables" → just the env-vars section).
  Sending only headers keeps the second call fast.

Why not embeddings / BM25?
  With a small, fixed set of overlapping docs the LLM understands query intent
  far better than vector similarity. "add a new domain" vs "add a new tool" are
  trivially different to any LLM but nearly identical in embedding space because
  they share almost all vocabulary.

To add a new doc: add one entry to DOC_REGISTRY below.
"""

import logging
import re
from pathlib import Path

from tools.shared import mcp, setup_logging, OLLAMA_BASE_URL, OLLAMA_MODEL

logger = setup_logging(__name__)


# ---------------------------------------------------------------------------
# Document registry
# One entry per file. The description is what the LLM sees when routing —
# write it as a concise answer to "what questions does this file answer?".
# ---------------------------------------------------------------------------

DOC_REGISTRY: dict[str, str] = {
    "Documentation/Security/auth-rbac.md": (
        "Atlas user authentication and access control: OIDC/Azure AD sign-in flow, "
        "session cookies, RBAC, group-based tool and category access"
    ),
    "Documentation/Security/chat-history-security.md": (
        "How chat history is stored, encrypted, and secured"
    ),
    "Documentation/Security/faqs.md": (
        "Security FAQs: prompt injection, RBAC bypass, session security"
    ),
    "Documentation/Integration/adding-a-tool.md": (
        "How to add a new MCP tool to an existing integration module "
        "(NetBrain, Panorama, Splunk, or any already-wired-up system)"
    ),
    "Documentation/Integration/adding-a-domain.md": (
        "How to connect a brand-new external system or API to Atlas from scratch: "
        "creating credentials in shared.py, writing the auth module, writing the tool module, "
        "registering in mcp_server.py"
    ),
    "Documentation/End-to-End-flow/netbrain.md": (
        "End-to-end request flow for NetBrain path queries: "
        "how a user question travels through Atlas to NetBrain and back"
    ),
    "Documentation/End-to-End-flow/panorama.md": (
        "End-to-end request flow for Panorama address group and object queries"
    ),
    "Documentation/End-to-End-flow/splunk.md": (
        "End-to-end request flow for Splunk firewall deny event queries"
    ),
    "Documentation/End-to-End-flow/faqs.md": (
        "FAQs about the end-to-end request flow"
    ),
    "Documentation/tools/netbrain.md": (
        "NetBrain MCP tools reference: API credentials, query pipeline, "
        "session token caching, path calculation, Panorama integration"
    ),
    "Documentation/tools/panorama.md": (
        "Panorama MCP tools reference: API credentials, address objects, "
        "address groups, security policies, caching, parallel fetching"
    ),
    "Documentation/tools/splunk.md": (
        "Splunk MCP tool reference: API credentials, firewall deny event search, "
        "query pipeline, event normalisation"
    ),
    "Documentation/General/operations.md": (
        "Operations and deployment guide: systemd service setup, how the .env file is loaded "
        "and which environment variables are required, starting and stopping Atlas servers"
    ),
    "Documentation/General/troubleshooting/troubleshooting.md": (
        "Troubleshooting guide: common errors, log locations, debugging steps"
    ),
    "Documentation/troubleshooting/troubleshooting.md": (
        "Troubleshooting guide: common errors, log locations, debugging steps"
    ),
}


# ---------------------------------------------------------------------------
# File reader — cached per process lifetime
# ---------------------------------------------------------------------------

_file_cache: dict[str, str] = {}


def _read_doc(rel_path: str) -> str | None:
    if rel_path in _file_cache:
        return _file_cache[rel_path]
    root = Path(__file__).resolve().parent.parent
    full = root / rel_path
    if not full.exists():
        logger.warning("Doc file not found: %s", full)
        return None
    text = full.read_text(encoding="utf-8")
    _file_cache[rel_path] = text
    return text


# ---------------------------------------------------------------------------
# Section splitter
# ---------------------------------------------------------------------------

def _split_sections(content: str) -> list[dict]:
    """Split markdown content on ## headings.

    Returns a list of dicts: {header: str | None, body: str}
    Header is None for any content that appears before the first ## heading.
    """
    sections: list[dict] = []
    parts = re.split(r"(?m)^(#{2,3} .+)$", content)
    # parts: [pre-heading, heading, body, heading, body, ...]
    intro = parts[0].strip()
    if intro:
        sections.append({"header": None, "body": intro})
    i = 1
    while i + 1 <= len(parts) - 1:
        header = parts[i].strip()
        body = parts[i + 1].strip()
        sections.append({"header": header, "body": body})
        i += 2
    return sections


# ---------------------------------------------------------------------------
# LLM-based file selection
# ---------------------------------------------------------------------------

def _pick_file(query: str) -> str | None:
    """Ask the LLM to pick the best file from the registry for this query."""
    from langchain_ollama import ChatOllama
    from langchain_core.messages import SystemMessage, HumanMessage

    registry_lines = "\n".join(
        f"- {path}: {desc}" for path, desc in DOC_REGISTRY.items()
    )

    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.0)
    messages = [
        SystemMessage(content=(
            "You are a documentation router. "
            "Return ONLY the file path (exactly as listed) of the single most relevant document. "
            "Do not explain. Do not add any other text. Output the file path only."
        )),
        HumanMessage(content=(
            f"Documents:\n{registry_lines}\n\n"
            f"Query: {query}\n\n"
            f"Most relevant file path:"
        )),
    ]

    try:
        response = llm.invoke(messages)
        chosen = response.content.strip().strip('"').strip("'")
    except Exception as exc:
        logger.error("LLM file selection failed: %s", exc)
        return None

    # Exact match
    if chosen in DOC_REGISTRY:
        return chosen

    # Partial match — model may include extra text or formatting
    for path in DOC_REGISTRY:
        if path in chosen:
            return path

    logger.warning("LLM returned unknown doc path %r for query %r", chosen, query)
    return None


# ---------------------------------------------------------------------------
# LLM-based section selection
# ---------------------------------------------------------------------------

def _pick_sections(content: str, query: str) -> str:
    """Return only the sections of *content* relevant to *query*.

    Sends just the section headings (not content) to the LLM and asks it
    which section numbers are needed. Falls back to the full file on any error.
    Files with ≤ 3 sections are returned whole — not worth the extra call.
    """
    from langchain_ollama import ChatOllama
    from langchain_core.messages import SystemMessage, HumanMessage

    sections = _split_sections(content)
    if len(sections) <= 3:
        return content

    def _preview(body: str) -> str:
        # First ~200 chars of body (across lines), stripped of markdown noise
        snippet = re.sub(r"\s+", " ", re.sub(r"[#`\[\]|*_\-]", "", body)).strip()[:200]
        return f" — {snippet}" if snippet else ""

    header_lines = "\n".join(
        f"{i}: {s['header'] or '(introduction / overview)'}{_preview(s['body'])}"
        for i, s in enumerate(sections)
    )

    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.0)
    messages = [
        SystemMessage(content=(
            "You are a documentation section selector. "
            "Given a numbered list of section headings and a user query, "
            "return the numbers of ALL sections needed to fully answer the query. "
            "Return ONLY comma-separated integers (e.g. '0,2,3'). "
            "No explanation. No other text."
        )),
        HumanMessage(content=(
            f"Sections:\n{header_lines}\n\n"
            f"Query: {query}\n\n"
            f"Relevant section numbers:"
        )),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()
    except Exception as exc:
        logger.error("LLM section selection failed: %s", exc)
        return content

    try:
        indices = sorted(set(
            int(x.strip()) for x in raw.replace(";", ",").split(",")
            if x.strip().lstrip("-").isdigit() and 0 <= int(x.strip()) < len(sections)
        ))
    except Exception:
        logger.warning("Could not parse section indices %r for query %r", raw, query)
        return content

    if not indices:
        logger.warning("LLM returned no valid section indices for query %r (raw: %r)", query, raw)
        return content

    logger.info(
        "Section selection: %d/%d sections for query %r → %s",
        len(indices), len(sections), query, indices,
    )

    parts = []
    for i in indices:
        s = sections[i]
        parts.append(f"{s['header']}\n\n{s['body']}" if s["header"] else s["body"])

    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# MCP tool
# ---------------------------------------------------------------------------

@mcp.tool()
def search_documentation(query: str) -> str:
    """Search the Atlas documentation to answer questions about how Atlas works.

    Use this tool when the user asks:
    - How a feature works (authentication, RBAC, sessions, tools, logging)
    - How to configure or deploy Atlas
    - How to troubleshoot a problem
    - What a specific component does
    - The end-to-end flow for a specific integration (NetBrain, Panorama, Splunk)
    - How to add a new tool or connect a new external system

    Args:
        query: The topic or question to search for.

    Returns:
        The most relevant documentation file as text.
        Returns a message if nothing relevant is found.
    """
    chosen = _pick_file(query)
    if not chosen:
        return "Could not determine the most relevant documentation file."

    content = _read_doc(chosen)
    if not content:
        return f"Documentation file not found: {chosen}"

    content = _pick_sections(content, query)

    logger.info("Serving doc: %s (query: %r)", chosen, query)
    return f"**Source: {chosen}**\n\n{content}"
