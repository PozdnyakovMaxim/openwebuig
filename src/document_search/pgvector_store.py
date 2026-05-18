from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlsplit, urlunsplit


DEFAULT_DATABASE_URL = ""


def database_url(value: str | None = None) -> str:
    return value or os.getenv("DATABASE_URL") or DEFAULT_DATABASE_URL


def redact_url(url: str) -> str:
    parts = urlsplit(url)
    if not parts.password:
        return url
    username = parts.username or ""
    hostname = parts.hostname or ""
    port = f":{parts.port}" if parts.port else ""
    netloc = f"{username}:***@{hostname}{port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _load_psycopg() -> tuple[Any, Any, Any]:
    try:
        import psycopg
        from psycopg.rows import dict_row
        from psycopg.types.json import Jsonb
    except ImportError as exc:
        raise RuntimeError("Missing dependency psycopg. Run `uv sync` or `pip install -e .`.") from exc
    return psycopg, dict_row, Jsonb


def connect(url: str | None = None) -> Any:
    psycopg, dict_row, _ = _load_psycopg()
    return psycopg.connect(database_url(url), row_factory=dict_row)


def vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{float(value):.8g}" for value in values) + "]"


def init_schema(conn: Any, *, embedding_dim: int, recreate: bool = False) -> list[str]:
    if embedding_dim <= 0 or embedding_dim > 4096:
        raise ValueError(f"Unexpected embedding dimension: {embedding_dim}")

    warnings: list[str] = []
    if recreate:
        conn.execute("DROP TABLE IF EXISTS doc_chunks")
        conn.execute("DROP TABLE IF EXISTS doc_documents")

    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS doc_documents (
            doc_id text PRIMARY KEY,
            source_name text NOT NULL,
            index_code text,
            document_title text,
            version text,
            metadata jsonb NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS doc_chunks (
            chunk_id text PRIMARY KEY,
            doc_id text NOT NULL REFERENCES doc_documents(doc_id) ON DELETE CASCADE,
            source_name text NOT NULL,
            index_code text,
            document_title text,
            version text,
            chunk_type text NOT NULL,
            citation_label text NOT NULL,
            raw_text text NOT NULL,
            searchable_text text NOT NULL,
            block_ids text[] NOT NULL,
            section_path text[] NOT NULL DEFAULT '{{}}',
            section_title text,
            subsection_title text,
            item_number text,
            appendix_number text,
            appendix_title text,
            char_count integer NOT NULL,
            embedding_model text NOT NULL,
            embedding vector({embedding_dim}) NOT NULL,
            metadata jsonb NOT NULL,
            search_tsv tsvector GENERATED ALWAYS AS (
                to_tsvector('russian', coalesce(searchable_text, ''))
            ) STORED,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS doc_chunks_doc_id_idx ON doc_chunks(doc_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS doc_chunks_source_idx ON doc_chunks(source_name)")
    conn.execute("CREATE INDEX IF NOT EXISTS doc_chunks_index_code_idx ON doc_chunks(index_code)")
    conn.execute("CREATE INDEX IF NOT EXISTS doc_chunks_search_tsv_idx ON doc_chunks USING gin(search_tsv)")
    conn.commit()

    try:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS doc_chunks_embedding_hnsw_idx "
            "ON doc_chunks USING hnsw (embedding vector_cosine_ops)"
        )
        conn.commit()
    except Exception as exc:
        conn.rollback()
        warnings.append(f"HNSW index was not created, falling back to IVFFLAT: {exc}")
        try:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS doc_chunks_embedding_ivfflat_idx "
                "ON doc_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
            )
            conn.commit()
        except Exception as fallback_exc:
            conn.rollback()
            warnings.append(f"IVFFLAT index was not created either: {fallback_exc}")

    return warnings


def upsert_document(conn: Any, metadata: dict[str, Any]) -> None:
    _, _, Jsonb = _load_psycopg()
    conn.execute(
        """
        INSERT INTO doc_documents (
            doc_id, source_name, index_code, document_title, version, metadata, updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, now())
        ON CONFLICT (doc_id) DO UPDATE SET
            source_name = EXCLUDED.source_name,
            index_code = EXCLUDED.index_code,
            document_title = EXCLUDED.document_title,
            version = EXCLUDED.version,
            metadata = EXCLUDED.metadata,
            updated_at = now()
        """,
        (
            metadata.get("doc_id"),
            metadata.get("source_name") or "",
            metadata.get("index_code"),
            metadata.get("display_title"),
            metadata.get("version"),
            Jsonb(metadata),
        ),
    )


def delete_chunks_for_doc(conn: Any, doc_id: str) -> None:
    conn.execute("DELETE FROM doc_chunks WHERE doc_id = %s", (doc_id,))


def upsert_chunk(conn: Any, chunk: dict[str, Any], *, embedding: list[float], embedding_model: str) -> None:
    _, _, Jsonb = _load_psycopg()
    conn.execute(
        """
        INSERT INTO doc_chunks (
            chunk_id, doc_id, source_name, index_code, document_title, version,
            chunk_type, citation_label, raw_text, searchable_text, block_ids, section_path,
            section_title, subsection_title, item_number, appendix_number, appendix_title,
            char_count, embedding_model, embedding, metadata, updated_at
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s::vector, %s, now()
        )
        ON CONFLICT (chunk_id) DO UPDATE SET
            source_name = EXCLUDED.source_name,
            index_code = EXCLUDED.index_code,
            document_title = EXCLUDED.document_title,
            version = EXCLUDED.version,
            chunk_type = EXCLUDED.chunk_type,
            citation_label = EXCLUDED.citation_label,
            raw_text = EXCLUDED.raw_text,
            searchable_text = EXCLUDED.searchable_text,
            block_ids = EXCLUDED.block_ids,
            section_path = EXCLUDED.section_path,
            section_title = EXCLUDED.section_title,
            subsection_title = EXCLUDED.subsection_title,
            item_number = EXCLUDED.item_number,
            appendix_number = EXCLUDED.appendix_number,
            appendix_title = EXCLUDED.appendix_title,
            char_count = EXCLUDED.char_count,
            embedding_model = EXCLUDED.embedding_model,
            embedding = EXCLUDED.embedding,
            metadata = EXCLUDED.metadata,
            updated_at = now()
        """,
        (
            chunk["chunk_id"],
            chunk["doc_id"],
            chunk["source_name"],
            chunk.get("index_code"),
            chunk.get("document_title"),
            chunk.get("version"),
            chunk["chunk_type"],
            chunk["citation_label"],
            chunk["raw_text"],
            chunk["searchable_text"],
            chunk.get("block_ids") or [],
            chunk.get("section_path") or [],
            chunk.get("section_title"),
            chunk.get("subsection_title"),
            chunk.get("item_number"),
            chunk.get("appendix_number"),
            chunk.get("appendix_title"),
            int(chunk.get("char_count") or len(chunk["raw_text"])),
            embedding_model,
            vector_literal(embedding),
            Jsonb(chunk),
        ),
    )


def count_rows(conn: Any) -> dict[str, int]:
    documents = conn.execute("SELECT count(*) AS value FROM doc_documents").fetchone()["value"]
    chunks = conn.execute("SELECT count(*) AS value FROM doc_chunks").fetchone()["value"]
    return {"documents": int(documents), "chunks": int(chunks)}


def sample_vector_search(conn: Any, *, embedding: list[float], limit: int = 5) -> list[dict[str, Any]]:
    return list(
        conn.execute(
            """
            SELECT
                chunk_id,
                doc_id,
                source_name,
                citation_label,
                raw_text,
                1 - (embedding <=> %s::vector) AS vector_score
            FROM doc_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (vector_literal(embedding), vector_literal(embedding), limit),
        ).fetchall()
    )
