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
        lines.append(f"{index}. {_document_label(document)}")
    if total > len(documents):
        lines.append(f"Показаны первые {len(documents)} из {total} документов.")
    return "\n".join(lines)


def documents_by_topic_answer(topic: str, documents: list[dict[str, object]]) -> str:
    if not documents:
        return f"В индексе не найдено документов по теме «{topic}»."

    lines = [f"Наиболее релевантные документы по теме «{topic}»:"]
    for index, document in enumerate(documents, start=1):
        lines.append(f"{index}. {_document_label(document)}")
    return "\n".join(lines)


def document_section_answer(section_reference: str, rows: list[dict[str, object]]) -> str:
    first = rows[0]
    title = str(first.get("document_title") or first.get("source_name") or "Без названия")
    texts = [str(row.get("raw_text") or "").strip() for row in rows]
    texts = [text for text in texts if text]
    return f"Пункт или раздел {section_reference} документа «{title}»:\n\n" + "\n\n".join(texts)


def document_section_not_found_answer(
    section_reference: str,
    *,
    document: dict[str, object] | None = None,
) -> str:
    if document:
        title = str(document.get("document_title") or document.get("source_name") or "Без названия")
        return f"В документе «{title}» не найден пункт или раздел {section_reference}."
    return f"В индексе не найден пункт или раздел {section_reference}."


def document_section_ambiguous_answer(
    section_reference: str,
    anchors: list[dict[str, object]],
) -> str:
    lines = [
        f"Ссылка {section_reference} найдена в нескольких местах. Уточните вариант:"
    ]
    for index, anchor in enumerate(anchors, start=1):
        title = _document_label(anchor)
        location: list[str] = []
        appendix_number = str(anchor.get("appendix_number") or "").strip()
        item_number = str(anchor.get("item_number") or "").strip()
        if appendix_number:
            location.append(f"приложение № {appendix_number}")
        elif anchor.get("section_labels"):
            labels = [str(value) for value in (anchor.get("section_labels") or [])]
            if labels:
                location.append(f"раздел {labels[-1]}")
        if item_number:
            location.append(f"пункт {item_number}")
        suffix = f" — {', '.join(location)}" if location else ""
        lines.append(f"{index}. {title}{suffix}")
    return "\n".join(lines)


def full_document_answer(
    document: dict[str, object],
    chunks: list[dict[str, object]],
    *,
    source_text: str | None = None,
) -> str:
    title = str(document.get("document_title") or document.get("source_name") or "Без названия")
    text = source_text or "\n\n".join(str(chunk.get("raw_text") or "").strip() for chunk in chunks)
    if not text.strip():
        return f"Документ «{title}» найден, но его текст отсутствует в индексе."

    return f"Полный текст документа «{title}»:\n\n{text.strip()}"


def document_not_found_answer(
    query: str,
    candidates: list[dict[str, object]],
) -> str:
    if not candidates:
        return f"Не удалось найти документ по запросу «{query}»."

    lines = [f"Не удалось однозначно определить документ по запросу «{query}». Возможные варианты:"]
    for index, document in enumerate(candidates, start=1):
        lines.append(f"{index}. {_document_label(document)}")
    lines.append("Уточните название, индекс или имя файла документа.")
    return "\n".join(lines)


def _document_label(document: dict[str, object]) -> str:
    title = str(document.get("document_title") or document.get("source_name") or "Без названия")
    qualifiers: list[str] = []
    source_name = str(document.get("source_name") or "").strip()
    index_code = str(document.get("index_code") or "").strip()
    version = str(document.get("version") or "").strip()
    if source_name and source_name != title:
        qualifiers.append(source_name)
    if index_code and index_code.casefold() not in title.casefold():
        qualifiers.append(f"индекс {index_code}")
    if version:
        qualifiers.append(f"версия {version}")
    return f"{title} ({'; '.join(qualifiers)})" if qualifiers else title


def _document_word(value: int) -> str:
    if value % 10 == 1 and value % 100 != 11:
        return "документ"
    if value % 10 in {2, 3, 4} and value % 100 not in {12, 13, 14}:
        return "документа"
    return "документов"
