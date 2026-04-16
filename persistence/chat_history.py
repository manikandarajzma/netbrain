"""
Persist chat history per user with multiple conversations (ChatGPT-style).

Storage backends (auto-selected based on REDIS_URL env var):
  1. Redis — when REDIS_URL is set. Encrypted blobs stored with AES-256-GCM.
  2. Disk  — fallback. Files at {APP_DIR}/data/chats/{sha256(username)}/.

Encryption at rest is mandatory for both backends: AES-256-GCM key from either
``CHAT_ENCRYPTION_KEY`` (local .env) or Azure Key Vault secret CHAT-ENCRYPTION-KEY
(when ``AZURE_KEYVAULT_URL`` is set).

Redis key structure:
  atlas:chats:{user_hash}:index          → encrypted index JSON
  atlas:chats:{user_hash}:conv:{id}      → encrypted conversation JSON
"""
import base64
import hashlib
import json
import logging
import os
import secrets
import uuid
from pathlib import Path
from typing import Any

_log = logging.getLogger("atlas.chat_history")

_MAX_MESSAGES_PER_CONV = 200
_MAX_TITLE_LEN = 60

# ---------------------------------------------------------------------------
# Encryption (shared by both backends)
# ---------------------------------------------------------------------------

_AES_KEY: bytes | None = None


def _get_aes_key() -> bytes:
    """Return the 32-byte AES-256 key loaded from Azure Key Vault."""
    global _AES_KEY
    if _AES_KEY is not None:
        return _AES_KEY

    # Local dev: CHAT_ENCRYPTION_KEY in .env (hex or base64, 32 bytes)
    local_key = os.getenv("CHAT_ENCRYPTION_KEY", "").strip()
    if local_key:
        if len(local_key) == 64 and all(c in "0123456789abcdefABCDEF" for c in local_key):
            _AES_KEY = bytes.fromhex(local_key)
        else:
            _AES_KEY = base64.b64decode(local_key)
        return _AES_KEY

    vault_url = os.getenv("AZURE_KEYVAULT_URL", "").strip().rstrip("/")
    if not vault_url:
        raise RuntimeError(
            "AZURE_KEYVAULT_URL is not set. "
            "Chat history encryption requires Azure Key Vault with secret CHAT-ENCRYPTION-KEY."
        )

    try:
        from azure.identity import DefaultAzureCredential
        from azure.keyvault.secrets import SecretClient
        cred = DefaultAzureCredential()
        client = SecretClient(vault_url=vault_url, credential=cred)
        s = client.get_secret("CHAT-ENCRYPTION-KEY")
        value = s.value if s and getattr(s, "value", None) else None
        if not value:
            raise RuntimeError("Key Vault secret CHAT-ENCRYPTION-KEY is missing or empty.")
        value = value.strip()
        if len(value) == 64 and all(c in "0123456789abcdefABCDEF" for c in value):
            key = bytes.fromhex(value)
        else:
            key = base64.b64decode(value)
        if len(key) != 32:
            raise RuntimeError(
                f"CHAT-ENCRYPTION-KEY must decode to exactly 32 bytes (got {len(key)}). "
                "Generate with: openssl rand -hex 32"
            )
        _AES_KEY = key
        return _AES_KEY
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError("Failed to load CHAT-ENCRYPTION-KEY from Azure Key Vault.") from e


def _encrypt(obj: dict | list) -> bytes:
    """Serialize and encrypt obj to base64(nonce + ciphertext)."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    plaintext = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    nonce = secrets.token_bytes(12)
    ct = AESGCM(_get_aes_key()).encrypt(nonce, plaintext, None)
    return base64.b64encode(nonce + ct)


def _decrypt(raw: bytes) -> dict | list:
    """Decrypt base64(nonce + ciphertext) and return parsed JSON."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    blob = base64.b64decode(raw)
    nonce, ct = blob[:12], blob[12:]
    plaintext = AESGCM(_get_aes_key()).decrypt(nonce, ct, None)
    return json.loads(plaintext.decode("utf-8"))


# ---------------------------------------------------------------------------
# Redis backend
# ---------------------------------------------------------------------------

_REDIS_CLIENT = None
_REDIS_CHECKED = False


def _get_redis():
    """Return a connected Redis client if REDIS_URL is set, else None."""
    global _REDIS_CLIENT, _REDIS_CHECKED
    if _REDIS_CHECKED:
        return _REDIS_CLIENT
    _REDIS_CHECKED = True
    url = os.getenv("REDIS_URL", "").strip()
    if not url:
        return None
    try:
        import redis
        client = redis.from_url(url, decode_responses=False)
        client.ping()
        _REDIS_CLIENT = client
        _log.info("Chat history using Redis backend: %s", url)
    except Exception as exc:
        _log.warning("Redis unavailable, falling back to disk chat history: %s", exc)
    return _REDIS_CLIENT


def _rkey_index(user_hash: str) -> str:
    return f"atlas:chats:{user_hash}:index"


def _rkey_conv(user_hash: str, conv_id: str) -> str:
    return f"atlas:chats:{user_hash}:conv:{conv_id}"


def _redis_get(key: str) -> dict | list | None:
    r = _get_redis()
    raw = r.get(key)
    if raw is None:
        return None
    try:
        return _decrypt(raw)
    except Exception:
        # Migration: try plaintext JSON
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return None


def _redis_set(key: str, obj: dict | list) -> None:
    r = _get_redis()
    r.set(key, _encrypt(obj))


def _redis_del(key: str) -> None:
    r = _get_redis()
    r.delete(key)


def _redis_list_conv_keys(user_hash: str) -> list[str]:
    r = _get_redis()
    pattern = f"atlas:chats:{user_hash}:conv:*"
    return [k.decode() if isinstance(k, bytes) else k for k in r.keys(pattern)]


# ---------------------------------------------------------------------------
# Disk backend (fallback)
# ---------------------------------------------------------------------------

_CHATS_DIR: Path | None = None


def _chats_dir(base_dir: Path) -> Path:
    global _CHATS_DIR
    if _CHATS_DIR is None:
        _CHATS_DIR = base_dir / "data" / "chats"
        _CHATS_DIR.mkdir(parents=True, exist_ok=True)
    return _CHATS_DIR


def _user_dir(base_dir: Path, username: str) -> Path:
    key = hashlib.sha256(username.encode("utf-8")).hexdigest()
    d = _chats_dir(base_dir) / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def _conv_path(user_dir: Path, conversation_id: str) -> Path:
    if not conversation_id or "/" in conversation_id or "\\" in conversation_id:
        raise ValueError("invalid conversation_id")
    return user_dir / f"{conversation_id}.json"


def _disk_read(path: Path) -> dict | list:
    """Read and decrypt a JSON file (AES-256-GCM), with plaintext fallback for migration."""
    raw = path.read_bytes()
    try:
        return _decrypt(raw)
    except Exception:
        try:
            result = json.loads(raw.decode("utf-8"))
            _log.warning("Plaintext chat file detected: %s — will re-encrypt on next write.", path)
            return result
        except Exception:
            raise ValueError(f"Could not decrypt or parse {path}")


def _disk_write(path: Path, obj: dict | list) -> None:
    path.write_bytes(_encrypt(obj))


def _disk_load_index(base_dir: Path, username: str) -> list[dict[str, Any]]:
    u = _user_dir(base_dir, username)
    path = u / "index.json"
    if not path.exists():
        return []
    try:
        data = _disk_read(path)
        entries = data.get("conversations", []) if isinstance(data, dict) else []
        return entries if isinstance(entries, list) else []
    except Exception:
        return []


def _disk_save_index(base_dir: Path, username: str, entries: list[dict[str, Any]]) -> None:
    u = _user_dir(base_dir, username)
    _disk_write(u / "index.json", {"conversations": entries})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _user_hash(username: str) -> str:
    return hashlib.sha256(username.encode("utf-8")).hexdigest()


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _truncate_title(first_message: str) -> str:
    if not first_message or not first_message.strip():
        return "New chat"
    t = first_message.strip().replace("\n", " ")[:_MAX_TITLE_LEN]
    return t + ("…" if len(first_message.strip()) > _MAX_TITLE_LEN else "")


# ---------------------------------------------------------------------------
# Public API — same signatures as before; Redis used when available
# ---------------------------------------------------------------------------

def list_conversations(base_dir: Path, username: str) -> list[dict[str, Any]]:
    """Return list of {id, title, created_at, parent_id?} in tree order."""
    if _get_redis() is not None:
        uh = _user_hash(username)
        data = _redis_get(_rkey_index(uh))
        entries = data.get("conversations", []) if isinstance(data, dict) else []
    else:
        entries = _disk_load_index(base_dir, username)

    roots = [e for e in entries if not e.get("parent_id")]
    roots.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    result: list[dict[str, Any]] = []
    for r in roots:
        result.append(dict(r))
        children = [e for e in entries if e.get("parent_id") == r.get("id")]
        children.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        for c in children:
            result.append(dict(c))
    return result


def get_conversation(base_dir: Path, username: str, conversation_id: str) -> list[dict[str, Any]] | None:
    """Return messages for the conversation, or None if not found."""
    if _get_redis() is not None:
        uh = _user_hash(username)
        data = _redis_get(_rkey_conv(uh, conversation_id))
        if data is None:
            return None
        messages = data.get("messages", []) if isinstance(data, dict) else []
        return messages if isinstance(messages, list) else []
    else:
        u = _user_dir(base_dir, username)
        path = _conv_path(u, conversation_id)
        if not path.exists():
            return None
        try:
            data = _disk_read(path)
            messages = data.get("messages", []) if isinstance(data, dict) else []
            return messages if isinstance(messages, list) else []
        except Exception:
            return None


def create_conversation(base_dir: Path, username: str, title: str, parent_id: str | None = None) -> str:
    """Create a new conversation; returns conversation_id."""
    conv_id = str(uuid.uuid4())
    entry: dict[str, Any] = {"id": conv_id, "title": title or "New chat", "created_at": _now()}
    if parent_id:
        entry["parent_id"] = parent_id

    if _get_redis() is not None:
        uh = _user_hash(username)
        data = _redis_get(_rkey_index(uh))
        entries = data.get("conversations", []) if isinstance(data, dict) else []
        entries.insert(0, entry)
        _redis_set(_rkey_index(uh), {"conversations": entries})
        _redis_set(_rkey_conv(uh, conv_id), {"messages": []})
    else:
        entries = _disk_load_index(base_dir, username)
        entries.insert(0, entry)
        _disk_save_index(base_dir, username, entries)
        u = _user_dir(base_dir, username)
        _disk_write(_conv_path(u, conv_id), {"messages": []})

    return conv_id


def append_to_conversation(
    base_dir: Path,
    username: str,
    conversation_id: str,
    user_message: str,
    assistant_content: Any,
) -> None:
    """Append one exchange to the conversation. Creates conversation if missing."""
    if _get_redis() is not None:
        uh = _user_hash(username)
        key = _rkey_conv(uh, conversation_id)
        data = _redis_get(key)
        if data is not None:
            messages = data.get("messages", []) if isinstance(data, dict) else []
        else:
            messages = []
            # Conversation missing from index — create entry
            idx_data = _redis_get(_rkey_index(uh))
            entries = idx_data.get("conversations", []) if isinstance(idx_data, dict) else []
            entries.insert(0, {
                "id": conversation_id,
                "title": _truncate_title(user_message),
                "created_at": _now(),
            })
            _redis_set(_rkey_index(uh), {"conversations": entries})
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": assistant_content})
        if len(messages) > _MAX_MESSAGES_PER_CONV:
            messages = messages[-_MAX_MESSAGES_PER_CONV:]
        _redis_set(key, {"messages": messages})
    else:
        u = _user_dir(base_dir, username)
        path = _conv_path(u, conversation_id)
        if path.exists():
            try:
                data = _disk_read(path)
                messages = data.get("messages", []) if isinstance(data, dict) else []
            except Exception:
                messages = []
        else:
            messages = []
            entries = _disk_load_index(base_dir, username)
            entries.insert(0, {
                "id": conversation_id,
                "title": _truncate_title(user_message),
                "created_at": _now(),
            })
            _disk_save_index(base_dir, username, entries)
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": assistant_content})
        if len(messages) > _MAX_MESSAGES_PER_CONV:
            messages = messages[-_MAX_MESSAGES_PER_CONV:]
        _disk_write(path, {"messages": messages})


def update_conversation_title(base_dir: Path, username: str, conversation_id: str, title: str) -> None:
    """Update the title of a conversation in the index."""
    if _get_redis() is not None:
        uh = _user_hash(username)
        data = _redis_get(_rkey_index(uh))
        entries = data.get("conversations", []) if isinstance(data, dict) else []
        for e in entries:
            if e.get("id") == conversation_id:
                e["title"] = (title or "New chat")[:_MAX_TITLE_LEN]
                break
        _redis_set(_rkey_index(uh), {"conversations": entries})
    else:
        entries = _disk_load_index(base_dir, username)
        for e in entries:
            if e.get("id") == conversation_id:
                e["title"] = (title or "New chat")[:_MAX_TITLE_LEN]
                break
        _disk_save_index(base_dir, username, entries)


def delete_conversation(base_dir: Path, username: str, conversation_id: str) -> bool:
    """Remove the conversation; returns True if it existed."""
    if _get_redis() is not None:
        uh = _user_hash(username)
        data = _redis_get(_rkey_index(uh))
        entries = data.get("conversations", []) if isinstance(data, dict) else []
        new_entries = [e for e in entries if e.get("id") != conversation_id]
        _redis_set(_rkey_index(uh), {"conversations": new_entries})
        deleted = _redis_client_delete(_rkey_conv(uh, conversation_id))
        return deleted > 0
    else:
        u = _user_dir(base_dir, username)
        entries = [e for e in _disk_load_index(base_dir, username) if e.get("id") != conversation_id]
        _disk_save_index(base_dir, username, entries)
        path = _conv_path(u, conversation_id)
        if path.exists():
            path.unlink()
            return True
        return False


def _redis_client_delete(key: str) -> int:
    r = _get_redis()
    return r.delete(key)


# ---------------------------------------------------------------------------
# Legacy: single conversation per user (backward compat)
# ---------------------------------------------------------------------------

def load_history(base_dir: Path, username: str) -> list[dict[str, Any]]:
    """Return messages from the most recent conversation, or []."""
    convs = list_conversations(base_dir, username)
    if not convs:
        return []
    return get_conversation(base_dir, username, convs[0]["id"]) or []


def append_to_history(base_dir: Path, username: str, user_message: str, assistant_content: Any) -> None:
    """Append to most recent conversation or create one."""
    convs = list_conversations(base_dir, username)
    if convs:
        append_to_conversation(base_dir, username, convs[0]["id"], user_message, assistant_content)
    else:
        conv_id = create_conversation(base_dir, username, _truncate_title(user_message))
        append_to_conversation(base_dir, username, conv_id, user_message, assistant_content)


def clear_history(base_dir: Path, username: str) -> None:
    """Delete all conversations for the user."""
    if _get_redis() is not None:
        uh = _user_hash(username)
        for key in _redis_list_conv_keys(uh):
            _get_redis().delete(key)
        _redis_del(_rkey_index(uh))
    else:
        u = _user_dir(base_dir, username)
        for e in _disk_load_index(base_dir, username):
            p = _conv_path(u, e["id"])
            if p.exists():
                p.unlink()
        _disk_save_index(base_dir, username, [])
