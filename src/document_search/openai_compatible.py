from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .pgvector_store import connect, count_rows, database_url
from .rag_service import answer_question, append_sources
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


@app.get("/health")
def health() -> dict[str, Any]:
    load_env_file()
    counts: dict[str, int] | None = None
    try:
        with connect(database_url()) as conn:
            counts = count_rows(conn)
    except Exception:
        counts = None
    return {"status": "ok", "model": _model_id(), "index": counts}


@app.get("/v1/models", response_model=ModelsResponse)
def list_models(authorization: str | None = Header(default=None)) -> ModelsResponse:
    _check_auth(authorization)
    return ModelsResponse(data=[ModelInfo(id=_model_id())])


@app.post("/v1/chat/completions", response_model=None)
def chat_completions(
    request: ChatCompletionRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any] | StreamingResponse:
    _check_auth(authorization)
    query = _last_user_text(request.messages)
    if not query:
        raise HTTPException(status_code=400, detail="No user message was provided.")

    limit = int(os.getenv("RAG_RETRIEVAL_LIMIT") or "6")
    force_extractive = _env_bool("RAG_FORCE_EXTRACTIVE", False)
    rag_answer = answer_question(
        query,
        limit=limit,
        extractive=force_extractive,
        temperature=float(request.temperature or 0.0),
    )
    content = append_sources(rag_answer)

    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    model = request.model or _model_id()
    if request.stream:
        return StreamingResponse(
            _stream_completion(response_id=response_id, created=created, model=model, content=content),
            media_type="text/event-stream",
        )

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


def _last_user_text(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return _content_to_text(message.content)
    if messages:
        return _content_to_text(messages[-1].content)
    return ""


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    parts.append(str(item["text"]))
                elif item.get("content"):
                    parts.append(str(item["content"]))
            elif item:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(content or "").strip()


def _stream_completion(*, response_id: str, created: int, model: str, content: str):
    first = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"
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
