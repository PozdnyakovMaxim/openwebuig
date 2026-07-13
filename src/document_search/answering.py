from __future__ import annotations

from typing import Any


SYSTEM_PROMPT = """Ты ГлавстройLLM, внутренний ассистент для помощи сотрудникам и специалистам ИТ-технической поддержки.
На текущий вопрос отвечай по предоставленным корпоративным документам.
Используй только предоставленные источники.
Если в источниках нет ответа, прямо скажи: "В найденных документах ответа нет."
Каждый существенный тезис подкрепляй ссылкой вида [1], [2].
Если вопрос уточняющий, используй историю диалога только для понимания текущего вопроса.
Не выдумывай номера разделов, даты, роли и обязанности."""


def format_sources(rows: list[dict[str, Any]], *, max_chars_per_source: int = 1400) -> str:
    parts: list[str] = []
    for index, row in enumerate(rows, start=1):
        text = " ".join(str(row["raw_text"]).split())
        if len(text) > max_chars_per_source:
            text = text[: max_chars_per_source - 1].rstrip() + "..."
        parts.append(
            "\n".join(
                [
                    f"[{index}] {row['citation_label']}",
                    text,
                ]
            )
        )
    return "\n\n".join(parts)


def format_chat_history(history: list[dict[str, str]], *, max_chars_per_message: int = 800) -> str:
    role_names = {"user": "Пользователь", "assistant": "Ассистент"}
    lines: list[str] = []
    for item in history:
        content = " ".join(str(item["content"]).split())
        if len(content) > max_chars_per_message:
            content = content[: max_chars_per_message - 1].rstrip() + "..."
        lines.append(f"{role_names.get(item['role'], item['role'])}: {content}")
    return "\n".join(lines)


def build_messages(
    query: str,
    rows: list[dict[str, Any]],
    *,
    chat_history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    sources = format_sources(rows)
    history = format_chat_history(chat_history or [])
    history_block = f"История диалога:\n{history}\n\n" if history else ""
    user_prompt = f"{history_block}Текущий вопрос: {query}\n\nИсточники:\n{sources}\n\nОтветь кратко, по делу, с цитатами."
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def extractive_answer(query: str, rows: list[dict[str, Any]], *, max_sources: int = 5) -> str:
    if not rows:
        return "В найденных документах ответа нет."

    lines = [f"Вопрос: {query}", "", "Ответ по найденным фрагментам:"]
    for index, row in enumerate(rows[:max_sources], start=1):
        text = " ".join(str(row["raw_text"]).split())
        if len(text) > 520:
            text = text[:519].rstrip() + "..."
        lines.append(f"- {text} [{index}]")

    lines.append("")
    lines.append("Источники:")
    for index, row in enumerate(rows[:max_sources], start=1):
        lines.append(f"[{index}] {row['citation_label']}")
    return "\n".join(lines)
