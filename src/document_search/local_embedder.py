from __future__ import annotations

import os
import threading
from typing import Any

from .settings import load_env_file


class LocalEmbedder:
    def __init__(
        self,
        *,
        engine: str | None = None,
        model: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        normalize: bool | None = None,
        use_fp16: bool | None = None,
        max_concurrency: int | None = None,
        index_id: str | None = None,
    ) -> None:
        load_env_file()
        self.engine = (engine or os.getenv("LOCAL_EMBED_ENGINE") or "sentence-transformers").lower()
        self.model = model or os.getenv("LOCAL_EMBED_MODEL") or ""
        self.device = device or os.getenv("LOCAL_EMBED_DEVICE") or None
        self.batch_size = (
            batch_size
            if batch_size is not None
            else int(os.getenv("LOCAL_EMBED_BATCH_SIZE") or "16")
        )
        self.normalize = normalize if normalize is not None else _env_bool("LOCAL_EMBED_NORMALIZE", True)
        self.use_fp16 = use_fp16 if use_fp16 is not None else _env_bool("LOCAL_EMBED_USE_FP16", True)
        self.max_concurrency = (
            max_concurrency
            if max_concurrency is not None
            else int(os.getenv("LOCAL_EMBED_MAX_CONCURRENCY") or "1")
        )
        base_index_id = index_id or os.getenv("LOCAL_EMBED_INDEX_ID") or f"local:{self.model}"
        profile_parts = [
            base_index_id,
            f"engine={self.engine}",
            f"normalize={str(self.normalize).lower()}",
        ]
        if self.engine in {"flagembedding", "flag"}:
            profile_parts.append(f"fp16={str(self.use_fp16).lower()}")
        self.index_id = "|".join(profile_parts)
        self._model: Any | None = None
        self._load_lock = threading.Lock()
        self._dimension: int | None = None
        self._dimension_lock = threading.Lock()

        if not self.model:
            raise ValueError("Local embedding model is not set. Use LOCAL_EMBED_MODEL or --embed-model.")
        if self.engine not in {"sentence-transformers", "flagembedding", "flag"}:
            raise ValueError("LOCAL_EMBED_ENGINE must be sentence-transformers or flagembedding.")
        if self.batch_size <= 0:
            raise ValueError("LOCAL_EMBED_BATCH_SIZE must be greater than zero.")
        if self.max_concurrency <= 0:
            raise ValueError("LOCAL_EMBED_MAX_CONCURRENCY must be greater than zero.")
        self._inference_gate = threading.BoundedSemaphore(self.max_concurrency)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with self._inference_gate:
            if self.engine == "sentence-transformers":
                return self._embed_sentence_transformers(texts)
            return self._embed_flagembedding(texts)

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embedding_dimension(self) -> int:
        if self._dimension is None:
            with self._dimension_lock:
                if self._dimension is None:
                    dimension = len(self.embed_text("dimension check"))
                    if dimension <= 0:
                        raise RuntimeError("Local embedding model returned an empty vector.")
                    self._dimension = dimension
        return self._dimension

    def _embed_sentence_transformers(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    try:
                        from sentence_transformers import SentenceTransformer
                    except ImportError as exc:
                        raise RuntimeError(
                            "Install sentence-transformers to use LOCAL_EMBED_ENGINE=sentence-transformers."
                        ) from exc
                    kwargs = {"device": self.device} if self.device else {}
                    self._model = SentenceTransformer(self.model, **kwargs)

        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def _embed_flagembedding(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    try:
                        from FlagEmbedding import BGEM3FlagModel
                    except ImportError as exc:
                        raise RuntimeError(
                            "Install FlagEmbedding to use LOCAL_EMBED_ENGINE=flagembedding."
                        ) from exc
                    self._model = BGEM3FlagModel(self.model, use_fp16=self.use_fp16)

        output = self._model.encode(texts, batch_size=self.batch_size)
        vectors = output["dense_vecs"] if isinstance(output, dict) else output
        return vectors.tolist() if hasattr(vectors, "tolist") else vectors


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}
