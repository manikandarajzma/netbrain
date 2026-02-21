"""
Shared infrastructure for MCP tool modules.

Owns the FastMCP instance, configuration constants, logging setup, and shared helpers.
Domain tool modules import `mcp` from here to register their tools.
"""

import os
import sys
import logging

# Ensure parent directory (netbrain/) is on sys.path so sibling modules
# (netbrainauth, panoramaauth) can be imported from tool modules.
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# ---------------------------------------------------------------------------
# Load .env file (must happen before any os.getenv calls)
# Look in netbrain/, project root, then cwd so one .env works everywhere.
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
        logger = setup_logging(__name__)   # e.g. "tools.netbox_tools"
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
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")


def _get_llm():
    """Get or initialize the LLM instance (lazy initialization)."""
    if mcp.llm is None:
        try:
            logger.debug("Initializing LLM (ChatOllama)...")
            llm = ChatOllama(
                model=OLLAMA_MODEL,
                temperature=0.0,
                base_url=OLLAMA_BASE_URL,
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

# NetBrain
NETBRAIN_URL = os.getenv("NETBRAIN_URL", "http://localhost")

# NetBox
NETBOX_URL = os.getenv("NETBOX_URL", "http://192.168.15.109:8080").rstrip("/")
NETBOX_TOKEN = os.getenv("NETBOX_TOKEN", "")
if not NETBOX_TOKEN:
    _vault_url = os.getenv("AZURE_KEYVAULT_URL", "").strip().rstrip("/")
    if _vault_url:
        _secret_name = os.getenv("NETBOX_TOKEN_KEYVAULT_SECRET_NAME", "NETBOX-TOKEN")
        for _name in (_secret_name, "netbox-token", "NetBoxToken"):
            try:
                from azure.identity import DefaultAzureCredential
                from azure.keyvault.secrets import SecretClient
                _credential = DefaultAzureCredential()
                _client = SecretClient(vault_url=_vault_url, credential=_credential)
                _secret = _client.get_secret(_name)
                if _secret and _secret.value:
                    NETBOX_TOKEN = _secret.value
                    logger.info("Loaded NETBOX_TOKEN from Azure Key Vault secret '%s'", _name)
                    break
            except Exception as e:
                if _name == _secret_name:
                    logger.warning(
                        "Key Vault: could not load secret '%s' from %s: %s. "
                        "Ensure this process has AZURE_CLIENT_ID/SECRET/TENANT_ID (or managed identity) and the app has 'Get' on secrets.",
                        _name, _vault_url, e,
                    )
                continue
        else:
            if not NETBOX_TOKEN:
                logger.warning("NETBOX_TOKEN not in env and not found in Key Vault (tried %s). NetBox tools will fail.", _secret_name)
NETBOX_VERIFY_SSL = os.getenv("NETBOX_VERIFY_SSL", "true").lower() in ["1", "true", "yes"]

# Splunk
SPLUNK_HOST = os.getenv("SPLUNK_HOST", "192.168.15.110")
SPLUNK_PORT = os.getenv("SPLUNK_PORT", "8089")
SPLUNK_USER = os.getenv("SPLUNK_USER", "")
SPLUNK_PASSWORD = os.getenv("SPLUNK_PASSWORD", "")

# MCP Server
MCP_SERVER_HOST = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8765"))
