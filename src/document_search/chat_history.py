from __future__ import annotations

import re
from typing import Any


CONTEXTUAL_QUERY_RE = re.compile(
    r"(?:\b(?:это|этот|эта|эти|там|тогда|он|она|они|его|её|их|такое|данн\w+|"
    r"перв\w+|втор\w+|трет\w+|предыдущ\w+)\b|^(?:а|и)\s+(?:кто|что|как|где|когда|почему)\b|"
    r"\b(?:подробнее|продолжи|уточни|поясни)\b)",
    re.IGNORECASE,
)
ORDINAL_REFERENCE_RE = re.compile(
    r"\b(?P<ordinal>перв\w*|втор\w*|трет\w*|четв[её]рт\w*|пят\w*|шест\w*|"
    r"седьм\w*|восьм\w*|девят\w*|десят\w*|\d{1,2}(?:-?[ыи]?[йяе])?)\s+"
    r"(?P<kind>пункт\w*|документ\w*|вариант\w*|источник\w*)\b",
    re.IGNORECASE,
)
NUMBERED_LIST_LINE_RE = re.compile(r"^\s*(?P<label>\d{1,3})[.)]\s+(?P<text>\S.+?)\s*$")
ORDINAL_STEMS = {
    "перв": 1,
    "втор": 2,
    "трет": 3,
    "четверт": 4,
    "четвёрт": 4,
    "пят": 5,
    "шест": 6,
    "седьм": 7,
    "восьм": 8,
    "девят": 9,
    "десят": 10,
}


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


def build_retrieval_query(
    query: str,
    history: list[dict[str, str]] | None = None,
    *,
    max_context_messages: int = 2,
    max_chars_per_message: int = 1800,
) -> str:
    current = query.strip()
    if not history or max_context_messages <= 0 or not CONTEXTUAL_QUERY_RE.search(current):
        return current

    context: list[str] = []
    for item in history[-max_context_messages:]:
        if item.get("role") not in {"user", "assistant"}:
            continue
        content = " ".join(str(item.get("content") or "").split())
        if not content:
            continue
        if len(content) > max_chars_per_message:
            marker = " ... "
            head = max_chars_per_message // 3
            tail = max_chars_per_message - head - len(marker)
            content = content[:head].rstrip() + marker + content[-tail:].lstrip()
        context.append(content)
    return "\n".join([*context, current])


def resolve_ordinal_reference(
    query: str,
    history: list[dict[str, str]] | None,
) -> dict[str, Any] | None:
    match = ORDINAL_REFERENCE_RE.search(query)
    if not match or not history:
        return None
    ordinal = _ordinal_number(match.group("ordinal"))
    if ordinal is None:
        return None

    for item in reversed(history):
        if item.get("role") != "assistant":
            continue
        numbered_lines = [
            line_match.groupdict()
            for line in str(item.get("content") or "").splitlines()
            if (line_match := NUMBERED_LIST_LINE_RE.match(line))
        ]
        if not numbered_lines:
            continue
        if len(numbered_lines) < ordinal:
            return None
        selected = numbered_lines[ordinal - 1]
        return {
            "position": ordinal,
            "kind": match.group("kind"),
            "list_label": selected["label"],
            "text": selected["text"],
        }
    return None


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


def _ordinal_number(value: str) -> int | None:
    digit_match = re.match(r"\d{1,2}", value)
    if digit_match:
        number = int(digit_match.group(0))
        return number if number > 0 else None
    lowered = value.lower().replace("ё", "е")
    for stem, number in ORDINAL_STEMS.items():
        if lowered.startswith(stem.replace("ё", "е")):
            return number
    return None
