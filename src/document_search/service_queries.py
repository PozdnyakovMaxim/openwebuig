from __future__ import annotations

import re


IDENTITY_PATTERNS = (
    r"кто\s+ты",
    r"ты\s+кто",
    r"что\s+ты\s+такое",
    r"как\s+тебя\s+зовут",
    r"представься",
)
CAPABILITY_PATTERNS = (
    r"что\s+ты\s+умеешь",
    r"что\s+можешь",
    r"чем\s+ты\s+можешь\s+помочь",
    r"какие\s+у\s+тебя\s+возможности",
)
DOCUMENT_PATTERNS = (
    r"(?:покажи|дай|выведи|перечисли)(?:\s+мне)?(?:\s+все)?(?:\s+доступные|\s+проиндексированные)?\s+документы",
    r"(?:список|перечень)\s+документов",
    r"(?:с\s+какими|по\s+каким|какие)\s+документ(?:ами|ы).*?(?:работаешь|доступны|есть\s+в\s+базе|проиндексированы)",
)


def classify_service_query(query: str) -> str | None:
    normalized = " ".join(query.lower().replace("ё", "е").split()).strip(" .!?;:")
    if any(re.fullmatch(pattern, normalized) for pattern in IDENTITY_PATTERNS):
        return "identity"
    if any(re.fullmatch(pattern, normalized) for pattern in CAPABILITY_PATTERNS):
        return "capabilities"
    if any(re.fullmatch(pattern, normalized) for pattern in DOCUMENT_PATTERNS):
        return "documents"
    return None


def identity_answer() -> str:
    return (
        "Я ГлавстройLLM, внутренний ассистент для помощи сотрудникам и специалистам "
        "ИТ-технической поддержки. Я нахожу информацию в проиндексированных корпоративных "
        "документах, отвечаю с указанием источников и сохраняю контекст текущего диалога."
    )


def capabilities_answer() -> str:
    return (
        "Я могу отвечать на вопросы по корпоративным документам, находить требования, роли, "
        "сроки и правила, продолжать уточняющий диалог и приводить источники к существенным "
        "утверждениям. Также я могу перечислить документы, которые сейчас находятся в индексе."
    )


def documents_answer(documents: list[dict[str, object]], *, total: int) -> str:
    if not documents:
        return "В индексе пока нет документов."

    lines = [f"Сейчас в индексе {total} {_document_word(total)}:"]
    for index, document in enumerate(documents, start=1):
        title = str(document.get("document_title") or document.get("source_name") or "Без названия")
        source_name = str(document.get("source_name") or "")
        if source_name and source_name != title:
            title = f"{title} ({source_name})"
        lines.append(f"{index}. {title}")
    if total > len(documents):
        lines.append(f"Показаны первые {len(documents)} из {total} документов.")
    return "\n".join(lines)


def _document_word(value: int) -> str:
    if value % 10 == 1 and value % 100 != 11:
        return "документ"
    if value % 10 in {2, 3, 4} and value % 100 not in {12, 13, 14}:
        return "документа"
    return "документов"
