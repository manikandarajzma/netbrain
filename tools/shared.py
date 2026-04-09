"""
Shared infrastructure for MCP tool modules.

Owns the FastMCP instance, configuration constants, logging setup, and shared helpers.
Domain tool modules import `mcp` from here to register their tools.
"""

import os
import sys
import logging

# Ensure parent directory (atlas/) is on sys.path so sibling modules
# (servicenowauth, etc.) can be imported from tool modules.
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# ---------------------------------------------------------------------------
# Load .env file (must happen before any os.getenv calls)
# Look in atlas/, project root, then cwd so one .env works everywhere.
# ---------------------------------------------------------------------------
from dotenv import load_dotenv
for _env_path in (
    os.path.join(_parent_dir, ".env"),
    os.path.join(os.path.dirname(_parent_dir), ".env"),
    os.path.join(os.getcwd(), ".env"),
):
    if os.path.isfile(_env_path):
        load_dotenv(_env_path)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()

def setup_logging(name: str) -> logging.Logger:
    """Create a named logger for a domain module.

    Usage in each module::

        from tools.shared import setup_logging
        logger = setup_logging(__name__)   # e.g. "tools.splunk_tools"
        logger.debug("something happened")
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.DEBUG))
    return logger

# Module-level logger for shared.py itself
logger = setup_logging("tools.shared")

# Disable SSL warnings from urllib3
import urllib3
urllib3.disable_warnings()

# ---------------------------------------------------------------------------
# FastMCP instance – the single server all domain modules register tools on
# ---------------------------------------------------------------------------
from fastmcp import FastMCP

mcp = FastMCP("atlas-mcp-server")

# LLM state (lazy-initialized)
mcp.llm = None
mcp.llm_error = None

# ---------------------------------------------------------------------------
# LLM helper (used by Panorama for AI analysis)
# ---------------------------------------------------------------------------
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate  # re-exported for convenience

# LLM configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4")


def _get_llm():
    """Get or initialize the LLM instance (lazy initialization)."""
    if mcp.llm is None:
        try:
            logger.debug("Initializing LLM (ChatOpenAI)...")
            llm = ChatOpenAI(
                model=OLLAMA_MODEL,
                temperature=0.0,
                base_url=OLLAMA_BASE_URL,
                api_key="docker",
            )
            mcp.llm = llm
            mcp.llm_error = None
            logger.debug("LLM initialized successfully")
        except Exception as e:
            error_msg = str(e)
            error_traceback = None
            try:
                import traceback
                error_traceback = traceback.format_exc()
            except:
                pass
            logger.error("LLM initialization failed: %s", error_msg)
            if error_traceback:
                logger.debug("LLM initialization traceback: %s", error_traceback)
            mcp.llm = False  # False = tried and failed (distinct from None = not tried)
            mcp.llm_error = {
                "error": error_msg,
                "traceback": error_traceback,
            }
    return mcp.llm if mcp.llm is not False else None

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# MCP Server
MCP_SERVER_HOST = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8765"))
