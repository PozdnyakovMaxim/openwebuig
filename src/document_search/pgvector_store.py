from __future__ import annotations

import hashlib
import os
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit


DEFAULT_DATABASE_URL = ""
CORPUS_PROMOTION_ADVISORY_LOCK_KEY = int.from_bytes(
    hashlib.sha256(b"openwebuig:corpus-promotion:v1").digest()[:8],
    byteorder="big",
    signed=True,
)


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


def acquire_corpus_read_lock(conn: Any) -> None:
    """Keep one corpus generation visible for the surrounding transaction."""

    conn.execute(
        "SELECT pg_advisory_xact_lock_shared(%s::bigint)",
        (CORPUS_PROMOTION_ADVISORY_LOCK_KEY,),
    )


def acquire_corpus_promotion_lock(conn: Any) -> None:
    """Serialize a complete corpus replacement against readers and writers."""

    conn.execute(
        "SELECT pg_advisory_xact_lock(%s::bigint)",
        (CORPUS_PROMOTION_ADVISORY_LOCK_KEY,),
    )


def resolve_embedding_index_id(embedder: Any) -> str:
    """Return the stable embedding profile ID stored alongside vectors."""

    value = getattr(embedder, "index_id", None) or getattr(embedder, "model", None)
    if callable(value):
        value = value()
    result = str(value or "").strip()
    if not result:
        raise ValueError("Embedder does not expose an index_id or model ID")
    return result


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
            section_labels text[] NOT NULL DEFAULT '{{}}',
            section_title text,
            subsection_title text,
            item_number text,
            heading_number text,
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
    # CREATE TABLE IF NOT EXISTS does not evolve an already deployed table.
    # Keep these migrations additive so upgrading a live index is safe.
    conn.execute(
        "ALTER TABLE doc_chunks "
        "ADD COLUMN IF NOT EXISTS section_labels text[] NOT NULL DEFAULT '{}'"
    )
    conn.execute("ALTER TABLE doc_chunks ADD COLUMN IF NOT EXISTS heading_number text")
    conn.execute("CREATE INDEX IF NOT EXISTS doc_chunks_doc_id_idx ON doc_chunks(doc_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS doc_chunks_source_idx ON doc_chunks(source_name)")
    conn.execute("CREATE INDEX IF NOT EXISTS doc_chunks_index_code_idx ON doc_chunks(index_code)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS doc_chunks_item_number_idx "
        "ON doc_chunks(item_number) WHERE item_number IS NOT NULL"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS doc_chunks_section_path_idx "
        "ON doc_chunks USING gin(section_path)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS doc_chunks_appendix_number_idx "
        "ON doc_chunks(appendix_number) WHERE appendix_number IS NOT NULL"
    )
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
            section_labels, section_title, subsection_title, item_number, heading_number,
            appendix_number, appendix_title, char_count, embedding_model, embedding, metadata,
            updated_at
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s::vector, %s, now()
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
            section_labels = EXCLUDED.section_labels,
            section_title = EXCLUDED.section_title,
            subsection_title = EXCLUDED.subsection_title,
            item_number = EXCLUDED.item_number,
            heading_number = EXCLUDED.heading_number,
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
            chunk.get("section_labels") or [],
            chunk.get("section_title"),
            chunk.get("subsection_title"),
            chunk.get("item_number"),
            chunk.get("heading_number"),
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


def list_documents(conn: Any, *, limit: int | None = None) -> tuple[list[dict[str, Any]], int]:
    total = int(conn.execute("SELECT count(*) AS value FROM doc_documents").fetchone()["value"])
    query = """
        SELECT source_name, document_title, index_code, version
        FROM doc_documents
        ORDER BY coalesce(document_title, source_name), source_name
        """
    parameters: tuple[Any, ...] = ()
    if limit is not None and limit > 0:
        query += " LIMIT %s"
        parameters = (limit,)
    rows = conn.execute(query, parameters).fetchall()
    return list(rows), total


def find_documents(conn: Any, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
    normalized = query.strip()
    if not normalized:
        return []
    return list(
        conn.execute(
            """
            SELECT
                doc_id,
                source_name,
                document_title,
                index_code,
                version,
                metadata,
                greatest(
                    similarity(lower(coalesce(document_title, '')), lower(%s)),
                    similarity(lower(source_name), lower(%s)),
                    similarity(lower(coalesce(index_code, '')), lower(%s))
                ) AS match_score
            FROM doc_documents
            ORDER BY
                CASE
                    WHEN lower(coalesce(document_title, '')) = lower(%s)
                        OR lower(source_name) = lower(%s)
                        OR lower(coalesce(index_code, '')) = lower(%s)
                    THEN 1 ELSE 0
                END DESC,
                match_score DESC,
                coalesce(document_title, source_name)
            LIMIT %s
            """,
            (normalized, normalized, normalized, normalized, normalized, normalized, limit),
        ).fetchall()
    )


def load_document_chunks(conn: Any, doc_id: str) -> list[dict[str, Any]]:
    return list(
        conn.execute(
            """
            SELECT
                chunk_id,
                doc_id,
                source_name,
                index_code,
                document_title,
                version,
                chunk_type,
                raw_text,
                citation_label,
                section_path,
                section_labels,
                section_title,
                subsection_title,
                item_number,
                heading_number,
                appendix_number,
                appendix_title
            FROM doc_chunks
            WHERE doc_id = %s AND chunk_type <> 'metadata'
            ORDER BY chunk_id
            """,
            (doc_id,),
        ).fetchall()
    )


def load_structural_chunks(
    conn: Any,
    section_reference: str,
    *,
    doc_id: str | None = None,
) -> list[dict[str, Any]]:
    """Load an exact item or a complete section subtree without semantic search."""

    reference = section_reference.strip()
    if not reference:
        return []

    appendix_match = re.fullmatch(
        r"(?:приложение|appendix)\s*(?:№|no\.?|#)?\s*(.+)",
        reference,
        re.IGNORECASE,
    )
    if appendix_match:
        normalized_reference = appendix_match.group(1).strip()
        return _load_structural_match(
            conn,
            predicate="appendix_number = %s",
            parameters=[normalized_reference],
            match_type="appendix",
            doc_id=doc_id,
        ) if normalized_reference else []

    item_match = re.fullmatch(
        r"(?:пункт|п\.)\s*(?:№|#)?\s*(.+)",
        reference,
        re.IGNORECASE,
    )
    if item_match:
        normalized_reference = item_match.group(1).strip()
        return _load_structural_match(
            conn,
            predicate="item_number = %s",
            parameters=[normalized_reference],
            match_type="item",
            doc_id=doc_id,
        ) if normalized_reference else []

    section_match = re.fullmatch(
        r"(?:раздел|section)\s*(?:№|#)?\s*(.+)",
        reference,
        re.IGNORECASE,
    )
    if section_match:
        normalized_reference = section_match.group(1).strip()
        return _load_structural_match(
            conn,
            predicate="(heading_number = %s OR section_path @> ARRAY[%s]::text[])",
            parameters=[normalized_reference, normalized_reference],
            match_type="section",
            doc_id=doc_id,
        ) if normalized_reference else []

    # Legacy/bare references prefer an exact item. Only if no item exists do
    # they fall back to a section subtree; this avoids merging unrelated anchors.
    normalized_reference = reference
    rows = _load_structural_match(
        conn,
        predicate="item_number = %s",
        parameters=[normalized_reference],
        match_type="item",
        doc_id=doc_id,
    )
    if rows:
        return rows
    return _load_structural_match(
        conn,
        predicate="(heading_number = %s OR section_path @> ARRAY[%s]::text[])",
        parameters=[normalized_reference, normalized_reference],
        match_type="section",
        doc_id=doc_id,
    )


def _load_structural_match(
    conn: Any,
    *,
    predicate: str,
    parameters: list[Any],
    match_type: str,
    doc_id: str | None,
) -> list[dict[str, Any]]:
    if match_type not in {"item", "section", "appendix"}:
        raise ValueError(f"Unsupported structural match type: {match_type}")
    query = f"""
        SELECT
            chunk_id,
            doc_id,
            source_name,
            index_code,
            document_title,
            version,
            chunk_type,
            citation_label,
            raw_text,
            block_ids,
            section_path,
            section_labels,
            section_title,
            subsection_title,
            item_number,
            heading_number,
            appendix_number,
            appendix_title,
            '{match_type}' AS structural_match
        FROM doc_chunks
        WHERE chunk_type <> 'metadata'
          AND {predicate}
    """
    if doc_id is not None:
        query += " AND doc_id = %s"
        parameters.append(doc_id)
    query += " ORDER BY doc_id, chunk_id"
    return list(conn.execute(query, tuple(parameters)).fetchall())


def validate_embedding_profile(
    conn: Any,
    *,
    expected_model: str,
    expected_dimension: int,
) -> dict[str, Any]:
    """Ensure the stored vectors match the query embedder before retrieval."""

    model_id = expected_model.strip()
    if not model_id:
        raise ValueError("expected_model must not be empty")
    if expected_dimension <= 0:
        raise ValueError("expected_dimension must be positive")

    rows = list(
        conn.execute(
            """
            SELECT
                embedding_model,
                vector_dims(embedding) AS embedding_dimension,
                count(*) AS chunk_count
            FROM doc_chunks
            GROUP BY embedding_model, vector_dims(embedding)
            ORDER BY embedding_model, embedding_dimension
            """
        ).fetchall()
    )
    if not rows:
        raise RuntimeError("Embedding index is empty; rebuild the corpus before querying it.")

    stored_models = sorted(
        {str(row.get("embedding_model") or "").strip() for row in rows}
    )
    stored_dimensions = sorted(
        {
            int(row["embedding_dimension"])
            for row in rows
            if row.get("embedding_dimension") is not None
        }
    )
    if "" in stored_models:
        raise RuntimeError("Embedding index contains chunks without an embedding model ID.")
    if len(stored_models) != 1:
        raise RuntimeError(
            "Embedding index contains mixed model IDs: " + ", ".join(stored_models)
        )
    if len(stored_dimensions) != 1:
        rendered = ", ".join(str(value) for value in stored_dimensions) or "unknown"
        raise RuntimeError(f"Embedding index contains mixed vector dimensions: {rendered}")

    stored_model = stored_models[0]
    stored_dimension = stored_dimensions[0]
    if stored_model != model_id:
        raise RuntimeError(
            "Embedding model mismatch: "
            f"index={stored_model!r}, query={model_id!r}. Rebuild the index or use the indexed model."
        )
    if stored_dimension != expected_dimension:
        raise RuntimeError(
            "Embedding dimension mismatch: "
            f"index={stored_dimension}, query={expected_dimension}. "
            "Rebuild the index or use the indexed model."
        )

    return {
        "model_id": stored_model,
        "dimension": stored_dimension,
        "chunks": sum(int(row.get("chunk_count") or 0) for row in rows),
    }


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
                section_path,
                section_labels,
                item_number,
                heading_number,
                1 - (embedding <=> %s::vector) AS vector_score
            FROM doc_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (vector_literal(embedding), vector_literal(embedding), limit),
        ).fetchall()
    )
