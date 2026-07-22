from __future__ import annotations

import json
import logging
import os
import secrets
import time
import uuid
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from .chat_history import content_to_text, normalize_history
from .external_document_loader import (
    DocumentTooLargeError,
    InvalidDocumentError,
    UnsupportedDocumentError,
    process_external_document,
)
from .openwebui_context import (
    OpenWebUIContext,
    build_openwebui_context_messages,
    clean_openwebui_history,
    parse_openwebui_context,
)
from .pgvector_store import (
    acquire_corpus_read_lock,
    connect,
    count_rows,
    database_url,
    resolve_embedding_index_id,
    validate_embedding_profile,
)
from .provider_api import make_chat, make_embedder
from .rag_service import RagAnswer, answer_question, append_sources, has_chat_config
from .settings import load_env_file


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float | None = 0.0
    max_tokens: int | None = None
    stream: bool = False


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "document-search"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


app = FastAPI(title="Document Search OpenAI-Compatible API")
logger = logging.getLogger("uvicorn.error")


@app.get("/health", response_model=None)
def health() -> dict[str, Any] | JSONResponse:
    load_env_file()
    try:
        embedder = make_embedder()
        expected_model = resolve_embedding_index_id(embedder)
        expected_dimension = int(os.getenv("RAG_EMBEDDING_DIM") or "1024")
        actual_dimension = int(embedder.embedding_dimension())
        if actual_dimension != expected_dimension:
            raise RuntimeError(
                "Configured embedding dimension does not match the query model: "
                f"configured={expected_dimension}, actual={actual_dimension}."
            )
        with connect(database_url()) as conn:
            acquire_corpus_read_lock(conn)
            counts = count_rows(conn)
            profile = validate_embedding_profile(
                conn,
                expected_model=expected_model,
                expected_dimension=actual_dimension,
            )
    except Exception as exc:
        logger.exception("readiness_check_failed")
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "model": _model_id(),
                "index": None,
                "error": "database_or_embedding_index_not_ready",
                "error_type": type(exc).__name__,
            },
        )
    return {
        "status": "ok",
        "model": _model_id(),
        "index": counts,
        "embedding_profile": profile,
    }


@app.get("/v1/models", response_model=ModelsResponse)
def list_models(authorization: str | None = Header(default=None)) -> ModelsResponse:
    _check_auth(authorization)
    return ModelsResponse(data=[ModelInfo(id=_model_id())])


@app.put("/process", response_model=None)
async def process_document(
    request: Request,
    authorization: str | None = Header(default=None),
    content_type: str = Header(default="", alias="Content-Type"),
    x_filename: str = Header(default="document.docx", alias="X-Filename"),
) -> dict[str, object]:
    """Open WebUI ExternalDocumentLoader endpoint.

    The official loader sends raw file bytes with PUT, Authorization,
    Content-Type and X-Filename headers and expects page_content/metadata JSON.
    """

    _check_document_loader_auth(authorization)
    max_bytes = _document_max_bytes()
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            declared_length = int(content_length)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header") from None
        if declared_length < 0:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header")
        if declared_length > max_bytes:
            raise HTTPException(status_code=413, detail="Uploaded document is too large")

    payload = bytearray()
    async for chunk in request.stream():
        if len(payload) + len(chunk) > max_bytes:
            raise HTTPException(status_code=413, detail="Uploaded document is too large")
        payload.extend(chunk)
    try:
        return await run_in_threadpool(
            process_external_document,
            bytes(payload),
            filename=x_filename,
            content_type=content_type,
            max_bytes=max_bytes,
        )
    except DocumentTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except UnsupportedDocumentError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except InvalidDocumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/chat/completions", response_model=None)
def chat_completions(
    request: ChatCompletionRequest,
    response: Response,
    authorization: str | None = Header(default=None),
) -> dict[str, Any] | StreamingResponse:
    _check_auth(authorization)
    query = _last_user_text(request.messages)
    if not query:
        raise HTTPException(status_code=400, detail="No user message was provided.")

    openwebui_context = parse_openwebui_context(request.messages)
    if openwebui_context is not None:
        rag_answer = _answer_openwebui_context(request, openwebui_context)
        content = rag_answer.answer
    else:
        limit = int(os.getenv("RAG_RETRIEVAL_LIMIT") or "6")
        chat_history_limit = int(os.getenv("RAG_CHAT_HISTORY_LIMIT") or "24")
        chat_history = normalize_history(request.messages, max_messages=chat_history_limit)
        force_extractive = _env_bool("RAG_FORCE_EXTRACTIVE", False)
        rag_answer = answer_question(
            query,
            limit=limit,
            chat_history=chat_history,
            extractive=force_extractive,
            temperature=float(request.temperature or 0.0),
        )
        content = append_sources(rag_answer)
    headers = _timing_headers(rag_answer.route, rag_answer.timings_ms)
    logger.info(
        "rag_request route=%s mode=%s timings_ms=%s",
        rag_answer.route,
        rag_answer.mode,
        json.dumps(rag_answer.timings_ms, ensure_ascii=False, sort_keys=True),
    )

    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    model = request.model or _model_id()
    if request.stream:
        return StreamingResponse(
            _stream_completion(response_id=response_id, created=created, model=model, content=content),
            media_type="text/event-stream",
            headers=headers,
        )

    for name, value in headers.items():
        response.headers[name] = value

    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "rag_metrics": {
            "route": rag_answer.route,
            "mode": rag_answer.mode,
            "timings_ms": rag_answer.timings_ms,
        },
    }


def _model_id() -> str:
    load_env_file()
    return os.getenv("OPENAI_COMPAT_MODEL_ID") or "document-search-rag"


def _check_auth(authorization: str | None) -> None:
    load_env_file()
    expected = os.getenv("OPENAI_COMPAT_API_KEY") or ""
    if not expected:
        return
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Invalid API key.")


def _check_document_loader_auth(authorization: str | None) -> None:
    load_env_file()
    expected = (
        os.getenv("OPENWEBUI_DOCUMENT_LOADER_API_KEY")
        or os.getenv("OPENAI_COMPAT_API_KEY")
        or ""
    )
    if not expected:
        raise HTTPException(status_code=503, detail="Document loader API key is not configured")
    supplied = authorization.removeprefix("Bearer ") if authorization else ""
    if not secrets.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Invalid API key.")


def _document_max_bytes() -> int:
    raw_value = os.getenv("OPENWEBUI_DOCUMENT_MAX_BYTES") or str(64 * 1024 * 1024)
    try:
        max_bytes = int(raw_value)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=503,
            detail="OPENWEBUI_DOCUMENT_MAX_BYTES must be a positive integer",
        ) from None
    if max_bytes <= 0:
        raise HTTPException(
            status_code=503,
            detail="OPENWEBUI_DOCUMENT_MAX_BYTES must be a positive integer",
        )
    return max_bytes


def _answer_openwebui_context(
    request: ChatCompletionRequest,
    context: OpenWebUIContext,
) -> RagAnswer:
    started = time.perf_counter()
    if not has_chat_config():
        raise HTTPException(status_code=503, detail="Chat provider is not configured")
    history = clean_openwebui_history(
        request.messages,
        max_messages=int(os.getenv("RAG_CHAT_HISTORY_LIMIT") or "24"),
    )
    messages = build_openwebui_context_messages(
        context,
        history=history,
        max_context_chars=int(os.getenv("OPENWEBUI_CONTEXT_MAX_CHARS") or "60000"),
    )
    chat = make_chat()
    generation_started = time.perf_counter()
    answer = chat.complete(
        messages,
        temperature=float(request.temperature or 0.0),
        max_tokens=request.max_tokens,
    )
    generation_ms = round((time.perf_counter() - generation_started) * 1000, 2)
    total_ms = round((time.perf_counter() - started) * 1000, 2)
    return RagAnswer(
        query=context.query,
        answer=answer,
        sources=[],
        rows=[],
        mode="generated",
        route="file_context",
        timings_ms={"generation": generation_ms, "total": total_ms},
    )


def _last_user_text(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return content_to_text(message.content)
    if messages:
        return content_to_text(messages[-1].content)
    return ""


def _stream_completion(*, response_id: str, created: int, model: str, content: str):
    for index, offset in enumerate(range(0, len(content), 2000)):
        delta: dict[str, str] = {"content": content[offset : offset + 2000]}
        if index == 0:
            delta["role"] = "assistant"
        chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    final = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _timing_headers(route: str, timings_ms: dict[str, float]) -> dict[str, str]:
    metrics = []
    for name in ("routing", "query", "embedding", "search", "database", "generation", "total"):
        if name in timings_ms:
            metrics.append(f"{name};dur={timings_ms[name]:.2f}")
    return {
        "Server-Timing": ", ".join(metrics),
        "X-RAG-Route": route,
        "X-RAG-Total-Ms": f"{timings_ms.get('total', 0.0):.2f}",
    }
