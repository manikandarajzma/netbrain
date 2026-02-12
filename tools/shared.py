"""
Shared infrastructure for MCP tool modules.

Owns the FastMCP instance, configuration constants, and shared helpers.
Domain tool modules import `mcp` from here to register their tools.
"""

import os
import sys

# Ensure parent directory (netbrain/) is on sys.path so sibling modules
# (netbrainauth, panoramaauth) can be imported from tool modules.
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# ---------------------------------------------------------------------------
# Load .env file (must happen before any os.getenv calls)
# ---------------------------------------------------------------------------
from dotenv import load_dotenv

# Look for .env in the netbrain/ directory (parent of tools/)
_env_path = os.path.join(_parent_dir, ".env")
if os.path.exists(_env_path):
    load_dotenv(_env_path)

# Disable SSL warnings from urllib3
import urllib3
urllib3.disable_warnings()

# ---------------------------------------------------------------------------
# FastMCP instance – the single server all domain modules register tools on
# ---------------------------------------------------------------------------
from fastmcp import FastMCP

mcp = FastMCP("netbrain-mcp-server")

# LLM state (lazy-initialized)
mcp.llm = None
mcp.llm_error = None

# ---------------------------------------------------------------------------
# LLM helper (used by NetBox and Panorama for AI analysis)
# ---------------------------------------------------------------------------
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate  # re-exported for convenience

# LLM configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")


def _get_llm():
    """Get or initialize the LLM instance (lazy initialization)."""
    if mcp.llm is None:
        try:
            print("DEBUG: Initializing LLM (ChatOllama)...", file=sys.stderr, flush=True)
            llm = ChatOllama(
                model=OLLAMA_MODEL,
                temperature=0.0,
                base_url=OLLAMA_BASE_URL,
            )
            mcp.llm = llm
            mcp.llm_error = None
            print("DEBUG: LLM initialized successfully", file=sys.stderr, flush=True)
        except Exception as e:
            error_msg = str(e)
            error_traceback = None
            try:
                import traceback
                error_traceback = traceback.format_exc()
            except:
                pass
            print(f"DEBUG: LLM initialization failed: {error_msg}", file=sys.stderr, flush=True)
            if error_traceback:
                print(f"DEBUG: LLM initialization traceback: {error_traceback}", file=sys.stderr, flush=True)
            mcp.llm = False  # False = tried and failed (distinct from None = not tried)
            mcp.llm_error = {
                "error": error_msg,
                "traceback": error_traceback,
            }
    return mcp.llm if mcp.llm is not False else None

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# NetBrain
NETBRAIN_URL = os.getenv("NETBRAIN_URL", "http://localhost")

# NetBox
NETBOX_URL = os.getenv("NETBOX_URL", "http://192.168.15.109:8080").rstrip("/")
NETBOX_TOKEN = os.getenv("NETBOX_TOKEN", "")
NETBOX_VERIFY_SSL = os.getenv("NETBOX_VERIFY_SSL", "true").lower() in ["1", "true", "yes"]

# Splunk
SPLUNK_HOST = os.getenv("SPLUNK_HOST", "192.168.15.110")
SPLUNK_PORT = os.getenv("SPLUNK_PORT", "8089")
SPLUNK_USER = os.getenv("SPLUNK_USER", "")
SPLUNK_PASSWORD = os.getenv("SPLUNK_PASSWORD", "")

# MCP Server
MCP_SERVER_HOST = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8765"))
