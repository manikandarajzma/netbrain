"""
Panorama API Authentication Module
Handles authentication with Palo Alto Panorama API using API key.

This module provides:
- API key-based authentication for Panorama REST API
- SSL certificate verification bypass for self-signed certificates
"""

# Import aiohttp for asynchronous HTTP client operations
import aiohttp

# Import json module for JSON serialization
import json

# Import logging for structured log output
import logging

# Import ssl for SSL/TLS context configuration
import ssl

# Import os module for accessing environment variables
import os

# Import urllib.parse for URL encoding
import urllib.parse

# Import xml.etree.ElementTree for parsing XML responses
import xml.etree.ElementTree as ET

# Import Optional type hint from typing module
from typing import Optional

# Load .env from this dir, project root, and cwd (same as tools/shared)
from dotenv import load_dotenv
_this_dir = os.path.dirname(os.path.abspath(__file__))
for _p in (
    os.path.join(_this_dir, ".env"),
    os.path.join(os.path.dirname(_this_dir), ".env"),
    os.path.join(os.getcwd(), ".env"),
):
    if os.path.isfile(_p):
        load_dotenv(_p)

# Configure module logger
logger = logging.getLogger("atlas.panoramaauth")

# Panorama API configuration
PANORAMA_URL = os.getenv("PANORAMA_URL", "https://192.168.15.247")

# Panorama username and password — always loaded from Azure Key Vault
PANORAMA_USERNAME = ""
PANORAMA_PASSWORD = ""
_vault_url = os.getenv("AZURE_KEYVAULT_URL", "").strip().rstrip("/")
_kv_tenant_id = os.getenv("AZURE_KEYVAULT_TENANT_ID", "").strip()
_kv_client_id = os.getenv("AZURE_KEYVAULT_CLIENT_ID", "").strip()
_kv_client_secret = os.getenv("AZURE_KEYVAULT_CLIENT_SECRET", "").strip()
if _vault_url and _kv_tenant_id and _kv_client_id and _kv_client_secret:
    try:
        from azure.identity import ClientSecretCredential
        from azure.keyvault.secrets import SecretClient
        _cred = ClientSecretCredential(
            tenant_id=_kv_tenant_id,
            client_id=_kv_client_id,
            client_secret=_kv_client_secret,
        )
        _client = SecretClient(vault_url=_vault_url, credential=_cred)
        _s = _client.get_secret("PANORAMA-USERNAME")
        if _s and _s.value:
            PANORAMA_USERNAME = _s.value
            logger.info("Loaded PANORAMA_USERNAME from Azure Key Vault")
        _s = _client.get_secret("PANORAMA-PASSWORD")
        if _s and _s.value:
            PANORAMA_PASSWORD = _s.value
            logger.info("Loaded PANORAMA_PASSWORD from Azure Key Vault")
    except Exception as e:
        logger.warning("Could not load Panorama credentials from Key Vault: %s", e)
else:
    logger.warning("AZURE_KEYVAULT_URL or KV service principal env vars not set; Panorama credentials unavailable")

async def get_api_key() -> Optional[str]:
    """
    Get Panorama API key using username/password authentication.
    Always requests a fresh key — no caching.

    Returns:
        Optional[str]: API key string if successful, None if authentication fails
    """
    # Validate that USERNAME and PASSWORD are configured
    if not PANORAMA_USERNAME or not PANORAMA_PASSWORD:
        logger.warning("Panorama USERNAME or PASSWORD not configured")
        return None

    # URL encode password to handle special characters
    password_encoded = urllib.parse.quote(PANORAMA_PASSWORD, safe='')

    # Construct the Panorama keygen API endpoint URL
    keygen_url = f"{PANORAMA_URL}/api/?type=keygen&user={PANORAMA_USERNAME}&password={password_encoded}"

    logger.debug(f"Attempting to get API key from: {PANORAMA_URL}")
    logger.debug(f"Username: {PANORAMA_USERNAME}")
    logger.debug(f"Keygen URL (password hidden): {PANORAMA_URL}/api/?type=keygen&user={PANORAMA_USERNAME}&password=***")

    # Create SSL context that doesn't verify certificates
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(keygen_url, ssl=ssl_context, timeout=10) as response:
                if response.status == 200:
                    # Parse XML response
                    response_text = await response.text()
                    try:
                        root = ET.fromstring(response_text)
                        status = root.attrib.get('status')

                        if status == 'success':
                            key_element = root.find('.//key')
                            if key_element is not None and key_element.text:
                                logger.debug("Panorama API key retrieved successfully")
                                return key_element.text
                            else:
                                logger.warning("API key not found in Panorama response")
                                return None
                        else:
                            # Extract error message if available
                            msg_element = root.find('.//msg')
                            error_msg = msg_element.text if msg_element is not None else "Unknown error"
                            logger.error(f"Panorama authentication failed: {error_msg}")
                            return None
                    except ET.ParseError as e:
                        logger.error(f"Error parsing Panorama XML response: {e}")
                        return None
                else:
                    logger.error(f"Get Panorama API key failed! HTTP {response.status}: {await response.text()}")
                    return None
    except Exception as e:
        logger.error(f"Error getting Panorama API key: {e}")
        return None
