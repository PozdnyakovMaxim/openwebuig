from __future__ import annotations

from dataclasses import dataclass, field
import os
import re
import time
from typing import Any

from .answering import build_messages, extractive_answer
from .chat_history import build_retrieval_query
from .document_content import load_source_document_text
from .pgvector_store import connect, database_url, find_documents, list_documents, load_document_chunks
from .provider_api import make_chat, make_embedder
from .query_router import RouteDecision, route_query
from .retriever import hybrid_search
from .service_queries import (
    capabilities_answer,
    document_not_found_answer,
    documents_answer,
    full_document_answer,
    identity_answer,
)
from .settings import load_env_file


@dataclass
class RagAnswer:
    query: str
    answer: str
    sources: list[str]
    rows: list[dict[str, Any]]
    mode: str
    route: str = "rag"
    timings_ms: dict[str, float] = field(default_factory=dict)


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
    started = time.perf_counter()
    load_env_file()
    chat = None
    routing_started = time.perf_counter()
    if has_chat_config(chat_provider, chat_model):
        chat = make_chat(
            provider=chat_provider,
            provider_api_base_url=provider_api_base_url,
            provider_api_key=provider_api_key,
            model=chat_model,
        )
        decision = route_query(chat, query, chat_history=chat_history)
    else:
        decision = RouteDecision(route="rag")
    routing_ms = _elapsed_ms(routing_started)
    service_route = decision.route

    if service_route == "identity":
        return _service_answer(query, identity_answer(), service_route, started, routing_ms=routing_ms)
    if service_route == "capabilities":
        return _service_answer(query, capabilities_answer(), service_route, started, routing_ms=routing_ms)
    if service_route == "documents":
        database_started = time.perf_counter()
        with connect(database_url(database_url_override)) as conn:
            documents, total = list_documents(conn)
        timings = {
            "routing": routing_ms,
            "database": _elapsed_ms(database_started),
            "total": _elapsed_ms(started),
        }
        return RagAnswer(
            query=query,
            answer=documents_answer(documents, total=total),
            sources=[],
            rows=[],
            mode="service",
            route="documents",
            timings_ms=timings,
        )
    if service_route == "full_document":
        database_started = time.perf_counter()
        document_query = decision.document_query or query
        with connect(database_url(database_url_override)) as conn:
            candidates = find_documents(conn, document_query)
            document = candidates[0] if candidates and float(candidates[0]["match_score"]) >= 0.2 else None
            chunks = load_document_chunks(conn, str(document["doc_id"])) if document else []
        source_text = load_source_document_text(document) if document else None
        timings = {
            "routing": routing_ms,
            "database": _elapsed_ms(database_started),
            "total": _elapsed_ms(started),
        }
        if document is None:
            content = document_not_found_answer(document_query, candidates)
        else:
            content = full_document_answer(document, chunks, source_text=source_text)
        return RagAnswer(
            query=query,
            answer=content,
            sources=[],
            rows=[],
            mode="service",
            route="full_document",
            timings_ms=timings,
        )
    if service_route == "general":
        return _service_answer(
            query,
            decision.answer,
            service_route,
            started,
            routing_ms=routing_ms,
            mode="generated",
        )

    retrieval_query_started = time.perf_counter()
    search_query = build_retrieval_query(query, chat_history)
    retrieval_query_ms = _elapsed_ms(retrieval_query_started)

    embedding_started = time.perf_counter()
    embedder = make_embedder(
        provider=embed_provider,
        provider_api_base_url=provider_api_base_url,
        provider_api_key=provider_api_key,
        model=embed_model,
    )
    embedding = embedder.embed_text(search_query)
    embedding_ms = _elapsed_ms(embedding_started)

    search_started = time.perf_counter()
    with connect(database_url(database_url_override)) as conn:
        rows = hybrid_search(conn, query=search_query, embedding=embedding, limit=limit)
    search_ms = _elapsed_ms(search_started)

    sources = [str(row["citation_label"]) for row in rows]
    if extractive or not has_chat_config(chat_provider, chat_model):
        answer_started = time.perf_counter()
        answer = extractive_answer(query, rows)
        return RagAnswer(
            query=query,
            answer=answer,
            sources=sources,
            rows=rows,
            mode="extractive",
            timings_ms={
                "routing": routing_ms,
                "query": retrieval_query_ms,
                "embedding": embedding_ms,
                "search": search_ms,
                "generation": _elapsed_ms(answer_started),
                "total": _elapsed_ms(started),
            },
        )

    if chat is None:
        chat = make_chat(
            provider=chat_provider,
            provider_api_base_url=provider_api_base_url,
            provider_api_key=provider_api_key,
            model=chat_model,
        )
    generation_started = time.perf_counter()
    answer = chat.complete(build_messages(query, rows, chat_history=chat_history), temperature=temperature)
    return RagAnswer(
        query=query,
        answer=answer,
        sources=sources,
        rows=rows,
        mode="generated",
        timings_ms={
            "routing": routing_ms,
            "query": retrieval_query_ms,
            "embedding": embedding_ms,
            "search": search_ms,
            "generation": _elapsed_ms(generation_started),
            "total": _elapsed_ms(started),
        },
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


def _service_answer(
    query: str,
    answer: str,
    route: str,
    started: float,
    *,
    routing_ms: float,
    mode: str = "service",
) -> RagAnswer:
    return RagAnswer(
        query=query,
        answer=answer,
        sources=[],
        rows=[],
        mode=mode,
        route=route,
        timings_ms={
            "routing": routing_ms,
            "total": _elapsed_ms(started),
        },
    )


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def _cited_source_numbers(text: str, *, max_source_number: int) -> list[int]:
    numbers: list[int] = []
    for match in re.finditer(r"\[(\d+)\]", text):
        number = int(match.group(1))
        if 1 <= number <= max_source_number and number not in numbers:
            numbers.append(number)
    return numbers
