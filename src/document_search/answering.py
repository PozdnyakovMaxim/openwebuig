from __future__ import annotations

from typing import Any


SYSTEM_PROMPT = """Ты отвечаешь на вопросы по предоставленным документам.
Используй только предоставленные источники.
Если в источниках нет ответа, прямо скажи: "В найденных документах ответа нет."
Каждый существенный тезис подкрепляй ссылкой вида [1], [2].
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


def build_messages(query: str, rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    sources = format_sources(rows)
    user_prompt = f"Вопрос: {query}\n\nИсточники:\n{sources}\n\nОтветь кратко, по делу, с цитатами."
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
