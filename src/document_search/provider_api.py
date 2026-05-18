from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .local_embedder import LocalEmbedder
from .settings import load_env_file


LOCAL_EMBEDDING_NAMES = {"local", "sentence-transformers", "flagembedding", "flag"}
PROVIDER_NAMES = {"provider", "api"}


class ProviderEmbedder:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int = 120,
        retries: int = 2,
    ) -> None:
        load_env_file()
        self.base_url = _normalize_provider_base_url(base_url or os.getenv("PROVIDER_API_BASE_URL") or "")
        self.api_key = api_key or os.getenv("PROVIDER_API_KEY") or ""
        self.model = model or os.getenv("PROVIDER_EMBED_MODEL") or ""
        self.timeout = timeout
        self.retries = retries
        if not self.base_url:
            raise ValueError("Provider API base URL is not set. Use PROVIDER_API_BASE_URL or --provider-api-base-url.")
        if not self.model:
            raise ValueError("Provider embedding model is not set. Use PROVIDER_EMBED_MODEL or --embed-model.")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {"model": self.model, "input": texts}
        data = self._post_json("/embeddings", payload)

        if isinstance(data.get("data"), list):
            items = sorted(data["data"], key=lambda item: item.get("index", 0))
            embeddings = [item.get("embedding") for item in items]
        else:
            embeddings = data.get("embeddings")

        if not isinstance(embeddings, list):
            raise RuntimeError(f"Provider did not return embeddings: {data}")
        if len(embeddings) != len(texts):
            raise RuntimeError(f"Provider returned {len(embeddings)} embeddings for {len(texts)} inputs.")
        for embedding in embeddings:
            if not isinstance(embedding, list) or not all(isinstance(value, int | float) for value in embedding):
                raise RuntimeError("Provider returned an invalid embedding payload.")
        return embeddings

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embedding_dimension(self) -> int:
        return len(self.embed_text("dimension check"))

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return _post_json(
            f"{self.base_url}{path}",
            payload,
            api_key=self.api_key,
            timeout=self.timeout,
            retries=self.retries,
        )


class ProviderChat:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int = 180,
        retries: int = 1,
    ) -> None:
        load_env_file()
        self.base_url = _normalize_provider_base_url(base_url or os.getenv("PROVIDER_API_BASE_URL") or "")
        self.api_key = api_key or os.getenv("PROVIDER_API_KEY") or ""
        self.model = model or os.getenv("PROVIDER_CHAT_MODEL") or ""
        self.timeout = timeout
        self.retries = retries
        if not self.base_url:
            raise ValueError("Provider API base URL is not set. Use PROVIDER_API_BASE_URL or --provider-api-base-url.")
        if not self.model:
            raise ValueError("Provider chat model is not set. Use PROVIDER_CHAT_MODEL or --chat-model.")

    def complete(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        data = self._post_json("/chat/completions", payload)
        choices = data.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content") or choices[0].get("text")
            if content:
                return str(content).strip()

        message = data.get("message") or {}
        if message.get("content"):
            return str(message["content"]).strip()
        raise RuntimeError(f"Provider chat did not return content: {data}")

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return _post_json(
            f"{self.base_url}{path}",
            payload,
            api_key=self.api_key,
            timeout=self.timeout,
            retries=self.retries,
        )


def make_embedder(
    *,
    provider: str | None = None,
    provider_api_base_url: str | None = None,
    provider_api_key: str | None = None,
    model: str | None = None,
) -> Any:
    load_env_file()
    selected = (provider or os.getenv("EMBEDDING_PROVIDER") or "local").lower()
    if selected in LOCAL_EMBEDDING_NAMES:
        engine = os.getenv("LOCAL_EMBED_ENGINE")
        if selected in {"sentence-transformers", "flagembedding", "flag"}:
            engine = selected
        return LocalEmbedder(engine=engine, model=model)
    if selected in PROVIDER_NAMES:
        return ProviderEmbedder(
            base_url=provider_api_base_url,
            api_key=provider_api_key,
            model=model,
        )
    raise ValueError(f"Unknown embedding provider: {selected}")


def make_chat(
    *,
    provider: str | None = None,
    provider_api_base_url: str | None = None,
    provider_api_key: str | None = None,
    model: str | None = None,
) -> Any:
    load_env_file()
    selected = (provider or os.getenv("CHAT_PROVIDER") or "provider").lower()
    if selected in PROVIDER_NAMES:
        return ProviderChat(
            base_url=provider_api_base_url,
            api_key=provider_api_key,
            model=model,
        )
    raise ValueError(f"Unknown chat provider: {selected}")


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    api_key: str,
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request = Request(url, data=body, headers=headers)
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code in {400, 401, 403, 404}:
                raise RuntimeError(f"Provider API returned HTTP {exc.code} for {url}") from exc
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Could not call provider API at {url}: {last_error}") from last_error


def _normalize_provider_base_url(raw_url: str) -> str:
    url = raw_url.rstrip("/")
    if not url:
        return ""

    parts = urlsplit(url)
    allow_http = os.getenv("ALLOW_INSECURE_PROVIDER_HTTP", "").strip().lower() in {"1", "true", "yes", "y", "on"}
    if parts.scheme == "http" and parts.hostname not in {"127.0.0.1", "localhost", "::1"} and not allow_http:
        raise ValueError(
            "Provider API with Bearer auth must use HTTPS unless it is localhost. "
            "Set ALLOW_INSECURE_PROVIDER_HTTP=true only for a trusted internal network."
        )

    path = parts.path.rstrip("/")
    if path in {"", "/"}:
        path = "/v1"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment)).rstrip("/")
