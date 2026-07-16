from __future__ import annotations

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


def full_document_answer(
    document: dict[str, object],
    chunks: list[dict[str, object]],
) -> str:
    title = str(document.get("document_title") or document.get("source_name") or "Без названия")
    if not chunks:
        return f"Документ «{title}» найден, но его текст отсутствует в индексе."

    text = "\n\n".join(str(chunk.get("raw_text") or "").strip() for chunk in chunks)
    return f"Полный текст документа «{title}»:\n\n{text.strip()}"


def document_not_found_answer(
    query: str,
    candidates: list[dict[str, object]],
) -> str:
    if not candidates:
        return f"Не удалось найти документ по запросу «{query}»."

    lines = [f"Не удалось однозначно определить документ по запросу «{query}». Возможные варианты:"]
    for index, document in enumerate(candidates, start=1):
        title = str(document.get("document_title") or document.get("source_name") or "Без названия")
        lines.append(f"{index}. {title}")
    lines.append("Уточните название документа.")
    return "\n".join(lines)


def _document_word(value: int) -> str:
    if value % 10 == 1 and value % 100 != 11:
        return "документ"
    if value % 10 in {2, 3, 4} and value % 100 not in {12, 13, 14}:
        return "документа"
    return "документов"
