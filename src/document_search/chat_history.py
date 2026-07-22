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
CITATION_LINE_RE = re.compile(r"^\s*\[(?P<label>\d{1,3})]\s+(?P<text>\S.+?)\s*$")
CITATION_REFERENCE_RE = re.compile(
    r"(?:"
    r"\b(?P<kind>источник\w*|ссылк\w*|цитат\w*|source\w*|citation\w*)\s*"
    r"(?:№|no\.?|#)?\s*\[?(?P<named>\d{1,3})\]?"
    r"|(?<![\w\]])\[(?P<bracket>\d{1,3})](?!\w)"
    r")",
    re.IGNORECASE,
)
ORDINAL_CITATION_REFERENCE_RE = re.compile(
    r"\b(?P<ordinal>перв\w*|втор\w*|трет\w*|четв[её]рт\w*|пят\w*|шест\w*|"
    r"седьм\w*|восьм\w*|девят\w*|десят\w*)\s+"
    r"(?P<kind>источник\w*|ссылк\w*|цитат\w*)\b",
    re.IGNORECASE,
)
IMPLICIT_CITATION_REFERENCE_RE = re.compile(
    r"\b(?:пункт\w*|подпункт\w*|раздел\w*|приложени\w*)\b"
    r"[^.!?\n]{0,160}\b(?:на\s+)?котор\w*\b[^.!?\n]{0,80}"
    r"\b(?:сослал\w*|ссыла\w*|процитировал\w*|цитировал\w*)\b",
    re.IGNORECASE,
)
EXPLICIT_STRUCTURAL_BRACKET_RE = re.compile(
    r"\b(?:пункт\w*|подпункт\w*|раздел\w*|приложени\w*|item|section|appendix)\s*"
    r"(?:№|#)?\s*\[\d{1,3}]",
    re.IGNORECASE,
)
ALL_LIST_SELECTION_RE = re.compile(
    r"^\s*(?:(?:покажи|выведи|выбери|давай|нужны?|беру)\s+)?"
    r"(?P<all>оба|обе|все|всё|both|all)"
    r"(?:\s+(?:вариант\w*|пункт\w*|документ\w*|источник\w*))?"
    r"(?:\s+(?:сразу|вместе))?[.!?]?\s*$",
    re.IGNORECASE,
)
STRUCTURAL_NUMBER_PATTERN = (
    r"(?:\d+(?:\.[0-9A-Za-zА-Яа-яЁё]+)*|[IVXLCDM]+|[A-Za-zА-Яа-яЁё])"
)
APPENDIX_ANCHOR_RE = re.compile(
    rf"\b(?:приложени\w*|appendix)\s*(?:№|no\.?|#)?\s*"
    rf"(?P<number>{STRUCTURAL_NUMBER_PATTERN})\b",
    re.IGNORECASE,
)
ITEM_ANCHOR_RE = re.compile(
    rf"\b(?:подпункт\w*|пункт\w*|п\.|item)\s*(?:№|#)?\s*"
    rf"(?P<number>{STRUCTURAL_NUMBER_PATTERN})\b",
    re.IGNORECASE,
)
SECTION_ANCHOR_RE = re.compile(
    rf"\b(?:раздел\w*|section)\s*(?:№|#)?\s*"
    rf"(?P<number>{STRUCTURAL_NUMBER_PATTERN})\b",
    re.IGNORECASE,
)
SOURCE_LABEL_PREFIX_RE = re.compile(
    r"^(?:источник|source)\s*\[\d{1,3}]\s*:\s*",
    re.IGNORECASE,
)
LIST_SELECTION_TRAILING_MODIFIERS_RE = re.compile(
    r"(?:подробнее|полностью|целиком|детальнее|отдельно|из\s+списка|из\s+вариантов)(?:\s+|$)",
    re.IGNORECASE,
)
LOWERCASE_SINGLE_LETTER_STOPWORDS = {"a", "i", "в", "и", "к", "о", "с"}
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

    if _ordinal_has_explicit_structural_tail(query, match):
        return None
    numbered_lines = _latest_numbered_lines(history)
    if len(numbered_lines) < ordinal:
        return None
    selected = numbered_lines[ordinal - 1]
    return {
        "position": ordinal,
        "kind": match.group("kind"),
        "list_label": selected["label"],
        "text": selected["text"],
    }


def resolve_dialog_reference(
    query: str,
    history: list[dict[str, str]] | None,
) -> dict[str, Any] | None:
    """Resolve an explicit source citation or a selection from the latest list.

    Source ordinals and document structure are deliberately separate namespaces:
    ``[2]`` and ``источник 2`` address the source list, while ``пункт 2`` is
    left untouched for the structural router.
    """

    citation = resolve_citation_reference(query, history)
    if citation is not None:
        return citation
    return resolve_numbered_list_selection(query, history)


def resolve_citation_reference(
    query: str,
    history: list[dict[str, str]] | None,
) -> dict[str, Any] | None:
    if not history:
        return None

    ordinal_match = ORDINAL_CITATION_REFERENCE_RE.search(query)
    match = _citation_reference_match(query)
    if ordinal_match:
        source_number = _ordinal_number(ordinal_match.group("ordinal"))
        kind = ordinal_match.group("kind")
    elif match:
        source_number = int(match.group("named") or match.group("bracket"))
        kind = match.group("kind") or "источник"
    elif IMPLICIT_CITATION_REFERENCE_RE.search(query):
        sources = _latest_citation_lines(history)
        if not sources:
            return None
        if len(sources) == 1:
            source_number = int(sources[0]["label"])
            kind = "источник"
        else:
            anchors = [
                anchor
                for source in sources
                if (anchor := parse_reference_anchor(source["text"])) is not None
            ]
            return {
                "reference_type": "citation_ambiguity",
                "source_numbers": [int(source["label"]) for source in sources],
                "items": [
                    {
                        "source_number": int(source["label"]),
                        "list_label": source["label"],
                        "text": source["text"],
                    }
                    for source in sources
                ],
                "anchors": anchors,
            }
    else:
        return None
    if source_number is None or source_number <= 0:
        return None

    source = _latest_labeled_line(history, CITATION_LINE_RE, str(source_number))
    text = source["text"] if source else ""
    anchor = parse_reference_anchor(text)
    return {
        "reference_type": "citation",
        "source_number": source_number,
        "position": source_number,
        "kind": kind,
        "list_label": source["label"] if source else str(source_number),
        "text": text,
        "found": source is not None,
        "anchors": [anchor] if anchor else [],
    }


def resolve_numbered_list_selection(
    query: str,
    history: list[dict[str, str]] | None,
) -> dict[str, Any] | None:
    """Resolve one or all choices from the most recent assistant numbered list."""

    if not history:
        return None
    all_match = ALL_LIST_SELECTION_RE.fullmatch(query)
    ordinal_match = ORDINAL_REFERENCE_RE.search(query)
    if not all_match and not ordinal_match:
        return None
    if all_match and all_match.group("all").casefold() in {"все", "всё", "all"}:
        if re.search(r"\b(?:документ\w*|источник\w*|documents?|sources?)\b", query, re.IGNORECASE):
            return None
    # ``второй источник`` belongs to the citation namespace, even if no source
    # with that number exists. Never reinterpret it as a numbered-list item.
    if ordinal_match and ordinal_match.group("kind").lower().startswith("источник"):
        return None

    numbered_lines = _latest_numbered_lines(history)
    if not numbered_lines:
        return None
    if all_match:
        selection_word = all_match.group("all").casefold()
        if selection_word in {"оба", "обе", "both"} and len(numbered_lines) != 2:
            return {
                "reference_type": "list_selection_ambiguity",
                "selection": "ambiguous_both",
                "requested_selection": selection_word,
                "available_count": len(numbered_lines),
                "positions": [index for index in range(1, len(numbered_lines) + 1)],
                "items": [
                    {
                        "position": index,
                        "list_label": line["label"],
                        "text": line["text"],
                    }
                    for index, line in enumerate(numbered_lines, start=1)
                ],
                "anchors": [],
            }
        selected = numbered_lines
        selection = "all"
        kind = "варианты"
    else:
        if _ordinal_has_explicit_structural_tail(query, ordinal_match):
            return None
        ordinal = _ordinal_number(ordinal_match.group("ordinal"))
        if ordinal is None or len(numbered_lines) < ordinal:
            return None
        selected = [numbered_lines[ordinal - 1]]
        selection = "single"
        kind = ordinal_match.group("kind")

    items: list[dict[str, Any]] = []
    anchors: list[dict[str, str]] = []
    for position, selected_line in enumerate(numbered_lines, start=1):
        if selected_line not in selected:
            continue
        item = {
            "position": position,
            "kind": kind,
            "list_label": selected_line["label"],
            "text": selected_line["text"],
        }
        items.append(item)
        anchor = parse_reference_anchor(selected_line["text"])
        if anchor:
            anchors.append(anchor)
    return {
        "reference_type": "list_selection",
        "selection": selection,
        "positions": [item["position"] for item in items],
        "items": items,
        "anchors": anchors,
    }


def parse_reference_anchor(text: str) -> dict[str, str] | None:
    """Extract document and structural fields from a rendered list/source label."""

    normalized = " ".join(text.split()).strip()
    if not normalized:
        return None
    appendix_match = APPENDIX_ANCHOR_RE.search(normalized)
    item_match = ITEM_ANCHOR_RE.search(normalized)
    section_match = SECTION_ANCHOR_RE.search(normalized)
    appendix_match = _validated_structural_match(appendix_match)
    item_match = _validated_structural_match(item_match)
    section_match = _validated_structural_match(section_match)

    structural_matches = [
        match for match in (appendix_match, item_match, section_match) if match is not None
    ]
    first_structural_start = min((match.start() for match in structural_matches), default=-1)
    document_query = ""
    if first_structural_start >= 0:
        document_query = normalized[:first_structural_start].rstrip(" ,;:—-")
    else:
        split = re.split(r"\s+[—–]\s+", normalized, maxsplit=1)
        if len(split) == 2:
            document_query = split[0].strip()
    document_query = SOURCE_LABEL_PREFIX_RE.sub("", document_query).strip()

    anchor: dict[str, str] = {"text": normalized}
    if document_query:
        anchor["document_query"] = document_query
    if appendix_match:
        anchor["appendix_number"] = appendix_match.group("number")
    if item_match:
        anchor["item_number"] = item_match.group("number")
    if section_match:
        anchor["section_number"] = section_match.group("number")

    if appendix_match and item_match:
        anchor["section_query"] = (
            f"приложение № {appendix_match.group('number')}, "
            f"пункт {item_match.group('number')}"
        )
    elif item_match:
        anchor["section_query"] = f"пункт {item_match.group('number')}"
    elif section_match:
        anchor["section_query"] = f"раздел {section_match.group('number')}"
    elif appendix_match:
        anchor["section_query"] = f"приложение № {appendix_match.group('number')}"

    return anchor if len(anchor) > 1 else None


def _latest_labeled_line(
    history: list[dict[str, str]],
    pattern: re.Pattern[str],
    requested_label: str,
) -> dict[str, str] | None:
    for item in reversed(history):
        if item.get("role") != "assistant":
            continue
        lines = [
            line_match.groupdict()
            for line in str(item.get("content") or "").splitlines()
            if (line_match := pattern.match(line))
        ]
        if not lines:
            continue
        return next((line for line in lines if line["label"] == requested_label), None)
    return None


def _latest_numbered_lines(history: list[dict[str, str]]) -> list[dict[str, str]]:
    for item in reversed(history):
        if item.get("role") != "assistant":
            continue
        return [
            line_match.groupdict()
            for line in str(item.get("content") or "").splitlines()
            if (line_match := NUMBERED_LIST_LINE_RE.match(line))
        ]
    return []


def _citation_reference_match(query: str) -> re.Match[str] | None:
    matches = list(CITATION_REFERENCE_RE.finditer(query))
    named_match = next((match for match in matches if match.group("named")), None)
    if named_match is not None:
        return named_match

    structural_spans = [match.span() for match in EXPLICIT_STRUCTURAL_BRACKET_RE.finditer(query)]
    for match in matches:
        if not match.group("bracket"):
            continue
        if any(start <= match.start() and match.end() <= end for start, end in structural_spans):
            continue
        return match
    return None


def _validated_structural_match(match: re.Match[str] | None) -> re.Match[str] | None:
    if match is None:
        return None
    number = match.group("number")
    if (
        len(number) == 1
        and number.isalpha()
        and number == number.casefold()
        and number.casefold() in LOWERCASE_SINGLE_LETTER_STOPWORDS
    ):
        return None
    return match


def _ordinal_has_explicit_structural_tail(query: str, match: re.Match[str]) -> bool:
    if not match.group("kind").casefold().startswith("пункт"):
        return False
    trailing = query[match.end() :].strip(" \t\r\n,.;:!?—–-")
    if not trailing:
        return False
    remainder = LIST_SELECTION_TRAILING_MODIFIERS_RE.sub("", trailing).strip()
    return bool(remainder)


def _latest_citation_lines(history: list[dict[str, str]]) -> list[dict[str, str]]:
    for item in reversed(history):
        if item.get("role") != "assistant":
            continue
        citation_lines = [
            line_match.groupdict()
            for line in str(item.get("content") or "").splitlines()
            if (line_match := CITATION_LINE_RE.match(line))
        ]
        if citation_lines:
            return citation_lines
    return []


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
