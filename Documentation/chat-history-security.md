# Securing chat history

Chat history is stored under `{APP_DIR}/data/chats/{sha256(username)}/` as JSON files (see `netbrain/chat_history.py`). To reduce risk for sensitive data (e.g. internal hostnames, IPs, or queries):

## 1. Encryption at rest (mandatory)

Chat history is always encrypted at rest. The app loads the Fernet key from Azure Key Vault secret **CHAT-ENCRYPTION-KEY**; **AZURE_KEYVAULT_URL** must be set. All index and conversation files are encrypted before write and decrypted on read. If the key is missing or Key Vault is unavailable, chat persistence fails at runtime.

The secret value must be a valid Fernet key (44-character base64). Generate one with:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Store that value in Azure Key Vault as the secret **CHAT-ENCRYPTION-KEY**. Use the same vault and access (e.g. managed identity or client credentials) as for other app secrets. The app will not persist or load chat history without this key.

## 2. File and directory permissions

- Run the app as a dedicated OS user and ensure only that user (and admins) can read the process and its files.
- Restrict the `data/chats/` directory so only the app user can read/write:
  - e.g. `chmod 700 data/chats` and `chmod 600 data/chats/*/*.json` (or rely on umask so new files are not world-readable).
- Keep `APP_DIR` (and any parent) off world-writable or overly permissive mounts.

## 3. Transport and access control

- Serve the app over **HTTPS** so chat traffic is encrypted in transit.
- Rely on **auth** (e.g. OIDC) so only authenticated users can call the chat API; chat files are keyed by hashed username so one user cannot read another’s files by path without being that user in the app.

## 4. Operational practices

- Back up the `data/chats/` tree and the Key Vault secret securely; losing the key means history cannot be decrypted.
- For high-sensitivity environments, consider stricter retention (e.g. periodic deletion or moving old files to a more controlled store) or additional redaction before persistence.
