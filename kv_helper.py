"""
Azure Key Vault helper — shared by all Atlas modules.

Provides a single get_secret() function using ClientSecretCredential
(AZURE_KEYVAULT_TENANT_ID / CLIENT_ID / CLIENT_SECRET env vars).
No DefaultAzureCredential — avoids slow managed-identity probes on non-Azure hosts.
"""
import os
from typing import Optional


def get_secret(secret_name: str) -> Optional[str]:
    """Fetch a secret value from Azure Key Vault.

    Returns the secret string, or None if the vault is not configured,
    the secret does not exist, or an error occurs.

    Required env vars:
        AZURE_KEYVAULT_URL
        AZURE_KEYVAULT_TENANT_ID
        AZURE_KEYVAULT_CLIENT_ID
        AZURE_KEYVAULT_CLIENT_SECRET
    """
    vault_url = os.getenv("AZURE_KEYVAULT_URL", "").strip().rstrip("/")
    tenant_id = os.getenv("AZURE_KEYVAULT_TENANT_ID", "").strip()
    client_id = os.getenv("AZURE_KEYVAULT_CLIENT_ID", "").strip()
    client_secret = os.getenv("AZURE_KEYVAULT_CLIENT_SECRET", "").strip()

    if not (vault_url and tenant_id and client_id and client_secret):
        return None

    try:
        from azure.identity import ClientSecretCredential
        from azure.keyvault.secrets import SecretClient
        cred = ClientSecretCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
        )
        client = SecretClient(vault_url=vault_url, credential=cred)
        s = client.get_secret(secret_name)
        return s.value if s and s.value else None
    except Exception:
        return None
