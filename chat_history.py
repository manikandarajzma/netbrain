"""
Persist chat history per user with multiple conversations (ChatGPT-style).
Location: {APP_DIR}/data/chats/{sha256(username)}/index.json and {id}.json per conversation.
"""
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

_CHATS_DIR: Path | None = None
_MAX_MESSAGES_PER_CONV = 200
_MAX_TITLE_LEN = 60


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
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = data.get("conversations", [])
        return entries if isinstance(entries, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_index(user_dir: Path, entries: list[dict[str, Any]]) -> None:
    _index_path(user_dir).write_text(
        json.dumps({"conversations": entries}, ensure_ascii=False),
        encoding="utf-8",
    )


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
        data = json.loads(path.read_text(encoding="utf-8"))
        messages = data.get("messages", [])
        return messages if isinstance(messages, list) else []
    except (json.JSONDecodeError, OSError):
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
    _conv_path(u, conv_id).write_text(json.dumps({"messages": []}, ensure_ascii=False), encoding="utf-8")
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
            data = json.loads(path.read_text(encoding="utf-8"))
            messages = data.get("messages", [])
        except (json.JSONDecodeError, OSError):
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
    path.write_text(json.dumps({"messages": messages}, ensure_ascii=False), encoding="utf-8")


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
