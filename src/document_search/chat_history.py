from __future__ import annotations

from typing import Any


def normalize_history(messages: list[Any], *, max_messages: int, exclude_last_user: bool = True) -> list[dict[str, str]]:
    if max_messages <= 0:
        return []

    history: list[dict[str, str]] = []
    for message in messages:
        role = _message_role(message)
        if role not in {"user", "assistant"}:
            continue
        content = content_to_text(_message_content(message))
        if content:
            history.append({"role": role, "content": content})

    if exclude_last_user:
        for index in range(len(history) - 1, -1, -1):
            if history[index]["role"] == "user":
                del history[index]
                break

    return history[-max_messages:]


def build_retrieval_query(query: str, history: list[dict[str, str]] | None = None, *, max_user_messages: int = 2) -> str:
    if not history or max_user_messages <= 0:
        return query

    user_messages = [item["content"] for item in history if item["role"] == "user" and item.get("content")]
    parts = [*user_messages[-max_user_messages:], query]
    return "\n".join(part.strip() for part in parts if part.strip())


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    parts.append(str(item["text"]))
                elif item.get("content"):
                    parts.append(str(item["content"]))
            elif item:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(content or "").strip()


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or "")
    return str(getattr(message, "role", "") or "")


def _message_content(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)
