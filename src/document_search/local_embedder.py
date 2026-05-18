from __future__ import annotations

import os
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
    ) -> None:
        load_env_file()
        self.engine = (engine or os.getenv("LOCAL_EMBED_ENGINE") or "sentence-transformers").lower()
        self.model = model or os.getenv("LOCAL_EMBED_MODEL") or ""
        self.device = device or os.getenv("LOCAL_EMBED_DEVICE") or None
        self.batch_size = batch_size or int(os.getenv("LOCAL_EMBED_BATCH_SIZE") or "16")
        self.normalize = normalize if normalize is not None else _env_bool("LOCAL_EMBED_NORMALIZE", True)
        self.use_fp16 = use_fp16 if use_fp16 is not None else _env_bool("LOCAL_EMBED_USE_FP16", True)
        self._model: Any | None = None

        if not self.model:
            raise ValueError("Local embedding model is not set. Use LOCAL_EMBED_MODEL or --embed-model.")
        if self.engine not in {"sentence-transformers", "flagembedding", "flag"}:
            raise ValueError("LOCAL_EMBED_ENGINE must be sentence-transformers or flagembedding.")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.engine == "sentence-transformers":
            return self._embed_sentence_transformers(texts)
        return self._embed_flagembedding(texts)

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embedding_dimension(self) -> int:
        return len(self.embed_text("dimension check"))

    def _embed_sentence_transformers(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("Install sentence-transformers to use LOCAL_EMBED_ENGINE=sentence-transformers.") from exc
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
            try:
                from FlagEmbedding import BGEM3FlagModel
            except ImportError as exc:
                raise RuntimeError("Install FlagEmbedding to use LOCAL_EMBED_ENGINE=flagembedding.") from exc
            self._model = BGEM3FlagModel(self.model, use_fp16=self.use_fp16)

        output = self._model.encode(texts, batch_size=self.batch_size)
        vectors = output["dense_vecs"] if isinstance(output, dict) else output
        return vectors.tolist() if hasattr(vectors, "tolist") else vectors


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}
