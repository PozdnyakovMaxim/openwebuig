from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Any

from .answering import build_messages, extractive_answer
from .chat_history import build_retrieval_query
from .pgvector_store import connect, database_url
from .provider_api import make_chat, make_embedder
from .retriever import hybrid_search
from .settings import load_env_file


@dataclass
class RagAnswer:
    query: str
    answer: str
    sources: list[str]
    rows: list[dict[str, Any]]
    mode: str


def has_chat_config(chat_provider: str | None = None, chat_model: str | None = None) -> bool:
    load_env_file()
    provider = (chat_provider or os.getenv("CHAT_PROVIDER") or "provider").lower()
    if chat_model:
        return True
    return provider == "provider" and bool(os.getenv("PROVIDER_CHAT_MODEL"))


def answer_question(
    query: str,
    *,
    database_url_override: str | None = None,
    embed_provider: str | None = None,
    provider_api_base_url: str | None = None,
    provider_api_key: str | None = None,
    embed_model: str | None = None,
    chat_provider: str | None = None,
    chat_model: str | None = None,
    chat_history: list[dict[str, str]] | None = None,
    limit: int = 6,
    extractive: bool = False,
    temperature: float = 0.0,
) -> RagAnswer:
    load_env_file()
    search_query = build_retrieval_query(query, chat_history)
    embedder = make_embedder(
        provider=embed_provider,
        provider_api_base_url=provider_api_base_url,
        provider_api_key=provider_api_key,
        model=embed_model,
    )
    embedding = embedder.embed_text(search_query)

    with connect(database_url(database_url_override)) as conn:
        rows = hybrid_search(conn, query=search_query, embedding=embedding, limit=limit)

    sources = [str(row["citation_label"]) for row in rows]
    if extractive or not has_chat_config(chat_provider, chat_model):
        return RagAnswer(
            query=query,
            answer=extractive_answer(query, rows),
            sources=sources,
            rows=rows,
            mode="extractive",
        )

    chat = make_chat(
        provider=chat_provider,
        provider_api_base_url=provider_api_base_url,
        provider_api_key=provider_api_key,
        model=chat_model,
    )
    return RagAnswer(
        query=query,
        answer=chat.complete(build_messages(query, rows, chat_history=chat_history), temperature=temperature),
        sources=sources,
        rows=rows,
        mode="generated",
    )


def append_sources(answer: RagAnswer) -> str:
    if not answer.sources or "Источники:" in answer.answer:
        return answer.answer

    cited_numbers = _cited_source_numbers(answer.answer, max_source_number=len(answer.sources))
    source_numbers = cited_numbers or list(range(1, len(answer.sources) + 1))

    lines = [answer.answer.rstrip(), "", "Источники:"]
    for index in source_numbers:
        lines.append(f"[{index}] {answer.sources[index - 1]}")
    return "\n".join(lines)


def _cited_source_numbers(text: str, *, max_source_number: int) -> list[int]:
    numbers: list[int] = []
    for match in re.finditer(r"\[(\d+)\]", text):
        number = int(match.group(1))
        if 1 <= number <= max_source_number and number not in numbers:
            numbers.append(number)
    return numbers
