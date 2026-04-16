"""
ServiceNow Authentication Module
Handles credentials loading from environment variables or Azure Key Vault.
Uses HTTP Basic Auth for the ServiceNow REST API.
"""

import os
from dotenv import load_dotenv

_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    load_dotenv(_env_path)

SERVICENOW_INSTANCE_URL = os.getenv("SERVICENOW_INSTANCE_URL", "https://dev252605.service-now.com")
SERVICENOW_USER = os.getenv("SERVICENOW_USER", "")
SERVICENOW_PASSWORD = os.getenv("SERVICENOW_PASSWORD", "")

if not SERVICENOW_USER or not SERVICENOW_PASSWORD:
    _vault_url = os.getenv("AZURE_KEYVAULT_URL", "").strip().rstrip("/")
    if _vault_url:
        try:
            from azure.identity import DefaultAzureCredential
            from azure.keyvault.secrets import SecretClient
            _credential = DefaultAzureCredential()
            _client = SecretClient(vault_url=_vault_url, credential=_credential)
            if not SERVICENOW_USER:
                try:
                    _secret = _client.get_secret("SERVICENOW-USER")
                    if _secret and _secret.value:
                        SERVICENOW_USER = _secret.value
                except Exception as e:
                    print(f"Key Vault: could not load secret 'SERVICENOW-USER': {e}")
            if not SERVICENOW_PASSWORD:
                try:
                    _secret = _client.get_secret("SERVICENOW-PASSWORD")
                    if _secret and _secret.value:
                        SERVICENOW_PASSWORD = _secret.value
                except Exception as e:
                    print(f"Key Vault: could not load secret 'SERVICENOW-PASSWORD': {e}")
        except Exception as e:
            print(f"Key Vault: failed to initialize client for ServiceNow credentials: {e}")
