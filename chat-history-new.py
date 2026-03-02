"""
Persist chat history per user with multiple conversations (ChatGPT-style).
Location: {APP_DIR}/data/chats/{sha256(username)}/index.json and {id}.json per conversation.

Encryption at rest is mandatory: the app loads a 32-byte AES-256 key from Azure Key Vault
secret CHAT-ENCRYPTION-KEY (AZURE_KEYVAULT_URL must be set). The secret value may be stored
as 64 hex characters or standard base64. All index and conversation files are encrypted with
AES-256-GCM before write and decrypted on read.

To generate a suitable key:
    openssl rand -hex 32
"""
import base64
import hashlib
import json
import os
import secrets
import uuid
from pathlib import Path
from typing import Any

from kv_helper import get_secret as _get_secret

_CHATS_DIR: Path | None = None
_AES_KEY: bytes | None = None  # None=uninited, else 32-byte AES-256 key
_MAX_MESSAGES_PER_CONV = 200
_MAX_TITLE_LEN = 60


import logging as _logging
_kv_logger = _logging.getLogger("atlas.chat_history")


def _decode_key(value: str) -> bytes:
    """Decode a hex or base64 string to a 32-byte AES key."""
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
    return key


def _get_aes_key() -> bytes:
    """Return the 32-byte AES-256 key.

    Load order:
      1. Azure Key Vault (secret CHAT-ENCRYPTION-KEY) via ClientSecretCredential —
         requires AZURE_KEYVAULT_URL + AZURE_KEYVAULT_TENANT_ID/CLIENT_ID/CLIENT_SECRET.
      2. CHAT_ENCRYPTION_KEY env var / .env fallback.
      3. Raise RuntimeError and log — never silently continue without a key.

    The key may be stored as 64 hex chars or standard base64.
    """
    global _AES_KEY
    if _AES_KEY is not None:
        return _AES_KEY

    # --- 1. Azure Key Vault ---
    value = _get_secret("CHAT-ENCRYPTION-KEY")
    if value:
        _AES_KEY = _decode_key(value)
        _kv_logger.info("Loaded CHAT-ENCRYPTION-KEY from Azure Key Vault")
        return _AES_KEY

    # --- 2. .env / environment variable fallback ---
    env_value = os.getenv("CHAT_ENCRYPTION_KEY", "").strip()
    if env_value:
        try:
            _AES_KEY = _decode_key(env_value)
            _kv_logger.info("Loaded CHAT-ENCRYPTION-KEY from environment variable")
            return _AES_KEY
        except Exception as e:
            _kv_logger.error("CHAT_ENCRYPTION_KEY env var is set but invalid: %s", e)

    # --- 3. Both sources failed ---
    msg = (
        "FAILED TO LOAD CHAT-ENCRYPTION-KEY: "
        "Set CHAT_ENCRYPTION_KEY in .env or configure Azure Key Vault with secret CHAT-ENCRYPTION-KEY. "
        "Generate a key with: openssl rand -hex 32"
    )
    _kv_logger.error(msg)
    raise RuntimeError(msg)


def _read_json_file(path: Path) -> dict[str, Any] | list[Any]:
    """Read and decrypt a JSON file (AES-256-GCM).

    File format: base64(nonce[12] || ciphertext+tag[len+16])

    Falls back to plaintext JSON if decryption fails, to allow one-time migration
    of pre-encryption files.
    """
    import logging
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    raw = path.read_bytes()
    try:
        blob = base64.b64decode(raw)
        nonce, ct = blob[:12], blob[12:]
        plaintext = AESGCM(_get_aes_key()).decrypt(nonce, ct, None)
        return json.loads(plaintext.decode("utf-8"))
    except Exception:
        # Migration fallback: try reading as plaintext JSON
        try:
            result = json.loads(raw.decode("utf-8"))
            logging.getLogger("atlas.chat_history").warning(
                "Plaintext (unencrypted) chat file detected: %s — will re-encrypt on next write.", path
            )
            return result
        except Exception:
            raise ValueError(f"Could not decrypt or parse {path}")


def _write_json_file(path: Path, obj: dict[str, Any] | list[Any]) -> None:
    """Encrypt and write JSON to file using AES-256-GCM.

    File format: base64(nonce[12] || ciphertext+tag)
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    plaintext = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    nonce = secrets.token_bytes(12)
    ct = AESGCM(_get_aes_key()).encrypt(nonce, plaintext, None)
    path.write_bytes(base64.b64encode(nonce + ct))


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


def _index_path(user_dir: Path) -> Path:
    return user_dir / "index.json"


def _conv_path(user_dir: Path, conversation_id: str) -> Path:
    if not conversation_id or "/" in conversation_id or "\\" in conversation_id:
        raise ValueError("invalid conversation_id")
    return user_dir / f"{conversation_id}.json"


def _load_index(user_dir: Path) -> list[dict[str, Any]]:
    path = _index_path(user_dir)
    if not path.exists():
        return []
    try:
        data = _read_json_file(path)
        entries = data.get("conversations", []) if isinstance(data, dict) else []
        return entries if isinstance(entries, list) else []
    except Exception:
        return []


def _save_index(user_dir: Path, entries: list[dict[str, Any]]) -> None:
    _write_json_file(_index_path(user_dir), {"conversations": entries})


def list_conversations(base_dir: Path, username: str) -> list[dict[str, Any]]:
    """Return list of { id, title, created_at, parent_id? } in tree order: root, then its children, then next root."""
    u = _user_dir(base_dir, username)
    entries = _load_index(u)
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
    u = _user_dir(base_dir, username)
    path = _conv_path(u, conversation_id)
    if not path.exists():
        return None
    try:
        data = _read_json_file(path)
        messages = data.get("messages", []) if isinstance(data, dict) else []
        return messages if isinstance(messages, list) else []
    except Exception:
        return None


def create_conversation(base_dir: Path, username: str, title: str, parent_id: str | None = None) -> str:
    """Create a new conversation with optional title and parent (for follow-ups); returns conversation_id."""
    conv_id = str(uuid.uuid4())
    u = _user_dir(base_dir, username)
    entries = _load_index(u)
    entry: dict[str, Any] = {"id": conv_id, "title": title or "New chat", "created_at": _now()}
    if parent_id:
        entry["parent_id"] = parent_id
    entries.insert(0, entry)
    _save_index(u, entries)
    _write_json_file(_conv_path(u, conv_id), {"messages": []})
    return conv_id


def append_to_conversation(
    base_dir: Path,
    username: str,
    conversation_id: str,
    user_message: str,
    assistant_content: Any,
) -> None:
    """Append one exchange to the conversation. Creates conversation if missing (legacy)."""
    u = _user_dir(base_dir, username)
    path = _conv_path(u, conversation_id)
    if path.exists():
        try:
            data = _read_json_file(path)
            messages = data.get("messages", []) if isinstance(data, dict) else []
        except (json.JSONDecodeError, OSError, TypeError):
            messages = []
    else:
        messages = []
        entries = _load_index(u)
        entries.insert(0, {"id": conversation_id, "title": _truncate_title(user_message), "created_at": _now()})
        _save_index(u, entries)
    messages.append({"role": "user", "content": user_message})
    messages.append({"role": "assistant", "content": assistant_content})
    if len(messages) > _MAX_MESSAGES_PER_CONV:
        messages = messages[-_MAX_MESSAGES_PER_CONV:]
    _write_json_file(path, {"messages": messages})


def update_conversation_title(base_dir: Path, username: str, conversation_id: str, title: str) -> None:
    """Update the title of a conversation in the index."""
    u = _user_dir(base_dir, username)
    entries = _load_index(u)
    for e in entries:
        if e.get("id") == conversation_id:
            e["title"] = (title or "New chat")[:_MAX_TITLE_LEN]
            break
    _save_index(u, entries)


def delete_conversation(base_dir: Path, username: str, conversation_id: str) -> bool:
    """Remove the conversation; returns True if it existed."""
    u = _user_dir(base_dir, username)
    path = _conv_path(u, conversation_id)
    entries = [e for e in _load_index(u) if e.get("id") != conversation_id]
    _save_index(u, entries)
    if path.exists():
        path.unlink()
        return True
    return False


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _truncate_title(first_message: str) -> str:
    if not first_message or not first_message.strip():
        return "New chat"
    t = first_message.strip().replace("\n", " ")[:_MAX_TITLE_LEN]
    return t + ("…" if len(first_message.strip()) > _MAX_TITLE_LEN else "")


# --- Legacy: single conversation per user (for backward compat / migration) ---

def load_history(base_dir: Path, username: str) -> list[dict[str, Any]]:
    """Load messages from the most recent conversation, or [] (backward compat)."""
    convs = list_conversations(base_dir, username)
    if not convs:
        return []
    return get_conversation(base_dir, username, convs[0]["id"]) or []


def append_to_history(base_dir: Path, username: str, user_message: str, assistant_content: Any) -> None:
    """Append to most recent conversation or create one (backward compat)."""
    convs = list_conversations(base_dir, username)
    if convs:
        append_to_conversation(base_dir, username, convs[0]["id"], user_message, assistant_content)
    else:
        conv_id = create_conversation(base_dir, username, _truncate_title(user_message))
        append_to_conversation(base_dir, username, conv_id, user_message, assistant_content)


def clear_history(base_dir: Path, username: str) -> None:
    """Delete all conversations for the user (backward compat)."""
    u = _user_dir(base_dir, username)
    for e in _load_index(u):
        p = _conv_path(u, e["id"])
        if p.exists():
            p.unlink()
    _save_index(u, [])
