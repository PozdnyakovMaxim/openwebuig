from __future__ import annotations

from dataclasses import dataclass, field
import os
import re
import time
from typing import Any

from .answering import build_messages, extractive_answer
from .chat_history import build_retrieval_query
from .document_content import load_source_document_text
from .document_ranking import rank_documents
from .pgvector_store import (
    acquire_corpus_read_lock,
    connect,
    database_url,
    find_documents,
    list_documents,
    load_document_chunks,
    load_structural_chunks,
    parse_structural_reference,
    resolve_embedding_index_id,
    validate_embedding_profile,
)
from .provider_api import make_chat, make_embedder
from .query_router import RouteDecision, route_query
from .retriever import hybrid_search
from .service_queries import (
    capabilities_answer,
    document_not_found_answer,
    document_section_answer,
    document_section_ambiguous_answer,
    document_section_not_found_answer,
    documents_answer,
    documents_by_topic_answer,
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
    if (decision.resolved_reference or {}).get("reference_type") == "citation_ambiguity":
        return _service_answer(
            query,
            _citation_ambiguity_answer(decision.resolved_reference or {}),
            "document_section",
            started,
            routing_ms=routing_ms,
        )
    if (decision.resolved_reference or {}).get("reference_type") == "list_selection_ambiguity":
        return _service_answer(
            query,
            _list_selection_ambiguity_answer(decision.resolved_reference or {}),
            "document_section",
            started,
            routing_ms=routing_ms,
        )
    structural_requests = _resolved_structural_requests(decision)
    if structural_requests and (decision.resolved_reference or {}).get("reference_type") == "list_selection":
        service_route = "document_section"

    if service_route == "identity":
        return _service_answer(query, identity_answer(), service_route, started, routing_ms=routing_ms)
    if service_route == "capabilities":
        return _service_answer(query, capabilities_answer(), service_route, started, routing_ms=routing_ms)
    if service_route == "documents":
        if decision.retrieval_query:
            return _answer_documents_by_topic(
                query,
                decision.retrieval_query,
                started=started,
                routing_ms=routing_ms,
                database_url_override=database_url_override,
                embed_provider=embed_provider,
                provider_api_base_url=provider_api_base_url,
                provider_api_key=provider_api_key,
                embed_model=embed_model,
            )
        database_started = time.perf_counter()
        with connect(database_url(database_url_override)) as conn:
            acquire_corpus_read_lock(conn)
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
    if service_route == "document_section":
        if not structural_requests:
            return _service_answer(
                query,
                decision.answer or "Уточните номер пункта или раздела.",
                service_route,
                started,
                routing_ms=routing_ms,
            )
        return _answer_document_sections(
            query,
            structural_requests,
            include_descendants=decision.include_descendants,
            database_url_override=database_url_override,
            started=started,
            routing_ms=routing_ms,
        )
    if service_route == "full_document":
        if not decision.document_query:
            return _service_answer(
                query,
                decision.answer or "Уточните, полный текст какого документа нужно вывести.",
                service_route,
                started,
                routing_ms=routing_ms,
            )
        database_started = time.perf_counter()
        document_query = decision.document_query
        with connect(database_url(database_url_override)) as conn:
            acquire_corpus_read_lock(conn)
            candidates = find_documents(conn, document_query)
            document = _select_document(candidates, document_query)
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
    search_query = decision.retrieval_query or build_retrieval_query(query, chat_history)
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
        acquire_corpus_read_lock(conn)
        validate_embedding_profile(
            conn,
            expected_model=resolve_embedding_index_id(embedder),
            expected_dimension=len(embedding),
        )
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


def _resolved_structural_requests(decision: RouteDecision) -> list[dict[str, str]]:
    reference = decision.resolved_reference or {}
    reference_type = str(reference.get("reference_type") or "")
    anchors = reference.get("anchors")
    selected: list[dict[str, Any]] = []
    if isinstance(anchors, list) and reference_type == "list_selection":
        selected = [anchor for anchor in anchors if isinstance(anchor, dict)]

    requests: list[dict[str, str]] = []
    for anchor in selected:
        section_query = str(anchor.get("section_query") or "").strip()
        if not section_query:
            continue
        requests.append(
            {
                "section_query": section_query,
                "document_query": str(anchor.get("document_query") or "").strip(),
            }
        )

    if (
        not requests
        and isinstance(anchors, list)
        and reference_type == "citation"
        and decision.route == "document_section"
    ):
        citation_anchor = next((anchor for anchor in anchors if isinstance(anchor, dict)), {})
        section_query = decision.section_query or str(
            citation_anchor.get("section_query") or ""
        ).strip()
        document_query = decision.document_query or str(
            citation_anchor.get("document_query") or ""
        ).strip()
        if section_query:
            requests.append(
                {
                    "section_query": section_query,
                    "document_query": document_query,
                }
            )

    if not requests and decision.route == "document_section" and decision.section_query:
        requests.append(
            {
                "section_query": decision.section_query,
                "document_query": decision.document_query,
            }
        )

    unique: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for request in requests:
        key = (
            request["document_query"].casefold(),
            request["section_query"].casefold(),
        )
        if key not in seen:
            seen.add(key)
            unique.append(request)
    return unique


def _citation_ambiguity_answer(reference: dict[str, Any]) -> str:
    items = reference.get("items")
    rendered: list[str] = []
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            source_number = item.get("source_number") or item.get("list_label")
            text = str(item.get("text") or "").strip()
            if source_number and text:
                rendered.append(f"{source_number}. Источник [{source_number}]: {text}")
    if not rendered:
        return "В предыдущем ответе было несколько источников. Уточните номер источника [N]."
    return "В предыдущем ответе было несколько ссылок. Уточните источник:\n" + "\n".join(rendered)


def _list_selection_ambiguity_answer(reference: dict[str, Any]) -> str:
    count = int(reference.get("available_count") or 0)
    if count > 0:
        return (
            f"В предыдущем списке {count} вариантов, поэтому слово «оба» неоднозначно. "
            "Укажите нужные номера или скажите «все»."
        )
    return "Не удалось однозначно определить два варианта. Укажите нужные номера."


def _answer_document_sections(
    query: str,
    requests: list[dict[str, str]],
    *,
    include_descendants: bool,
    database_url_override: str | None,
    started: float,
    routing_ms: float,
) -> RagAnswer:
    database_started = time.perf_counter()
    content_blocks: list[str] = []
    successful_rows: list[dict[str, Any]] = []
    with connect(database_url(database_url_override)) as conn:
        acquire_corpus_read_lock(conn)
        for request in requests:
            section_query = request["section_query"]
            document_query = request["document_query"]
            candidates: list[dict[str, Any]] = []
            document: dict[str, Any] | None = None
            if document_query:
                candidates = find_documents(conn, document_query)
                document = _select_document(candidates, document_query)
                rows = (
                    load_structural_chunks(
                        conn,
                        section_query,
                        doc_id=str(document["doc_id"]),
                        include_descendants=include_descendants,
                    )
                    if document
                    else []
                )
            else:
                rows = load_structural_chunks(
                    conn,
                    section_query,
                    include_descendants=include_descendants,
                )

            if document_query and document is None:
                content_blocks.append(document_not_found_answer(document_query, candidates))
                continue

            structural_anchors = _distinct_structural_anchors(
                rows,
                section_query,
                include_descendants=include_descendants,
            )
            if len(structural_anchors) > 1:
                content_blocks.append(
                    document_section_ambiguous_answer(section_query, structural_anchors)
                )
            elif not rows:
                content_blocks.append(
                    document_section_not_found_answer(section_query, document=document)
                )
            else:
                content_blocks.append(document_section_answer(section_query, rows))
                successful_rows.extend(rows)

    rows = _deduplicate_rows(successful_rows)
    timings = {
        "routing": routing_ms,
        "database": _elapsed_ms(database_started),
        "total": _elapsed_ms(started),
    }
    return RagAnswer(
        query=query,
        answer="\n\n---\n\n".join(content_blocks),
        sources=_unique_citations(rows),
        rows=rows,
        mode="service",
        route="document_section",
        timings_ms=timings,
    )


def _deduplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        key = str(row.get("chunk_id") or "")
        if not key:
            key = "\x1f".join(
                str(row.get(field) or "")
                for field in ("doc_id", "citation_label", "raw_text", "item_number")
            ) or str(index)
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def _answer_documents_by_topic(
    query: str,
    topic_query: str,
    *,
    started: float,
    routing_ms: float,
    database_url_override: str | None,
    embed_provider: str | None,
    provider_api_base_url: str | None,
    provider_api_key: str | None,
    embed_model: str | None,
) -> RagAnswer:
    embedding_started = time.perf_counter()
    embedder = make_embedder(
        provider=embed_provider,
        provider_api_base_url=provider_api_base_url,
        provider_api_key=provider_api_key,
        model=embed_model,
    )
    embedding = embedder.embed_text(topic_query)
    embedding_ms = _elapsed_ms(embedding_started)

    search_started = time.perf_counter()
    with connect(database_url(database_url_override)) as conn:
        acquire_corpus_read_lock(conn)
        validate_embedding_profile(
            conn,
            expected_model=resolve_embedding_index_id(embedder),
            expected_dimension=len(embedding),
        )
        rows = hybrid_search(
            conn,
            query=topic_query,
            embedding=embedding,
            limit=100,
            vector_candidates=180,
            text_candidates=180,
        )
    documents = rank_documents(
        rows,
        limit=int(os.getenv("RAG_DOCUMENT_RESULT_LIMIT") or "12"),
        min_vector_score=float(os.getenv("RAG_DOCUMENT_MIN_VECTOR_SCORE") or "0.45"),
        min_text_score=float(os.getenv("RAG_DOCUMENT_MIN_TEXT_SCORE") or "0.01"),
        min_document_score=float(os.getenv("RAG_DOCUMENT_MIN_SCORE") or "0.05"),
        relative_cutoff=float(os.getenv("RAG_DOCUMENT_RELATIVE_CUTOFF") or "0.55"),
        min_competitive_signal=float(
            os.getenv("RAG_DOCUMENT_MIN_COMPETITIVE_SIGNAL") or "0.25"
        ),
    )
    timings = {
        "routing": routing_ms,
        "embedding": embedding_ms,
        "search": _elapsed_ms(search_started),
        "total": _elapsed_ms(started),
    }
    return RagAnswer(
        query=query,
        answer=documents_by_topic_answer(topic_query, documents),
        sources=[],
        rows=[],
        mode="service",
        route="documents",
        timings_ms=timings,
    )


def _select_document(
    candidates: list[dict[str, Any]],
    query: str,
) -> dict[str, Any] | None:
    if not candidates:
        return None
    normalized_query = query.strip().casefold()
    for field in ("source_name", "index_code", "document_title"):
        exact = [
            candidate
            for candidate in candidates
            if str(candidate.get(field) or "").strip().casefold() == normalized_query
        ]
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            return None
    for field in ("source_name", "index_code", "document_title"):
        contained = [
            candidate
            for candidate in candidates
            if (value := str(candidate.get(field) or "").strip().casefold())
            and value in normalized_query
        ]
        if len(contained) == 1:
            return contained[0]
    first_score = float(candidates[0].get("match_score") or 0.0)
    if first_score < 0.2:
        return None
    if len(candidates) > 1:
        second_score = float(candidates[1].get("match_score") or 0.0)
        if first_score - second_score < 0.035:
            return None
    return candidates[0]


def _distinct_structural_anchors(
    rows: list[dict[str, Any]],
    section_reference: str,
    *,
    include_descendants: bool = False,
) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        doc_id = str(row.get("doc_id") or "")
        if not doc_id:
            continue
        match_type = str(row.get("structural_match") or "")
        if match_type == "item":
            if include_descendants:
                reference = parse_structural_reference(section_reference)
                root_number = reference.number if reference else section_reference.casefold()
                location = (
                    row.get("appendix_number"),
                    tuple(row.get("section_path") or []),
                    "item_subtree",
                    root_number,
                )
            else:
                block_ids = tuple(str(value) for value in (row.get("block_ids") or []))
                location = (
                    row.get("appendix_number"),
                    tuple(row.get("section_path") or []),
                    row.get("item_number"),
                    block_ids,
                )
        elif match_type == "appendix":
            location = (row.get("appendix_number"),)
        elif match_type == "section":
            location = (row.get("appendix_number"), section_reference.casefold())
        else:
            # Compatibility for mocked/older projections: preserve the former
            # one-anchor-per-document behavior when match provenance is absent.
            location = ("document",)
        key = (doc_id, match_type, *location)
        if key in seen:
            continue
        seen.add(key)
        anchors.append(row)
    return anchors


def _unique_citations(rows: list[dict[str, Any]]) -> list[str]:
    citations: list[str] = []
    for row in rows:
        value = str(row.get("citation_label") or "").strip()
        if value and value not in citations:
            citations.append(value)
    return citations


def _cited_source_numbers(text: str, *, max_source_number: int) -> list[int]:
    numbers: list[int] = []
    for match in re.finditer(r"\[(\d+)\]", text):
        number = int(match.group(1))
        if 1 <= number <= max_source_number and number not in numbers:
            numbers.append(number)
    return numbers
