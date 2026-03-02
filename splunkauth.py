"""
Splunk Authentication Module
Handles Splunk credentials loading from environment variables or Azure Key Vault.
"""

import os
from dotenv import load_dotenv

_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    load_dotenv(_env_path)

SPLUNK_HOST = os.getenv("SPLUNK_HOST", "192.168.15.110")
SPLUNK_PORT = os.getenv("SPLUNK_PORT", "8089")
SPLUNK_USER = os.getenv("SPLUNK_USER", "")
SPLUNK_PASSWORD = os.getenv("SPLUNK_PASSWORD", "")

if not SPLUNK_USER or not SPLUNK_PASSWORD:
    _vault_url = os.getenv("AZURE_KEYVAULT_URL", "").strip().rstrip("/")
    if _vault_url:
        try:
            from azure.identity import DefaultAzureCredential
            from azure.keyvault.secrets import SecretClient
            _credential = DefaultAzureCredential()
            _client = SecretClient(vault_url=_vault_url, credential=_credential)
            if not SPLUNK_USER:
                try:
                    _secret = _client.get_secret("SPLUNK-USER")
                    if _secret and _secret.value:
                        SPLUNK_USER = _secret.value
                except Exception as e:
                    print(f"Key Vault: could not load secret 'SPLUNK-USER' from {_vault_url}: {e}")
            if not SPLUNK_PASSWORD:
                try:
                    _secret = _client.get_secret("SPLUNK-PASSWORD")
                    if _secret and _secret.value:
                        SPLUNK_PASSWORD = _secret.value
                except Exception as e:
                    print(f"Key Vault: could not load secret 'SPLUNK-PASSWORD' from {_vault_url}: {e}")
        except Exception as e:
            print(f"Key Vault: failed to initialize client for Splunk credentials: {e}")
