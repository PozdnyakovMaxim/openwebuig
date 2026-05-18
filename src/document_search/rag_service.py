from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from .answering import build_messages, extractive_answer
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
    limit: int = 6,
    extractive: bool = False,
    temperature: float = 0.0,
) -> RagAnswer:
    load_env_file()
    embedder = make_embedder(
        provider=embed_provider,
        provider_api_base_url=provider_api_base_url,
        provider_api_key=provider_api_key,
        model=embed_model,
    )
    embedding = embedder.embed_text(query)

    with connect(database_url(database_url_override)) as conn:
        rows = hybrid_search(conn, query=query, embedding=embedding, limit=limit)

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
        answer=chat.complete(build_messages(query, rows), temperature=temperature),
        sources=sources,
        rows=rows,
        mode="generated",
    )


def append_sources(answer: RagAnswer) -> str:
    if not answer.sources or "Источники:" in answer.answer:
        return answer.answer

    lines = [answer.answer.rstrip(), "", "Источники:"]
    for index, source in enumerate(answer.sources, start=1):
        lines.append(f"[{index}] {source}")
    return "\n".join(lines)
