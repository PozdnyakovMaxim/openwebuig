from __future__ import annotations

from typing import Any

from .pgvector_store import vector_literal


def hybrid_search(
    conn: Any,
    *,
    query: str,
    embedding: list[float],
    limit: int = 8,
    vector_candidates: int = 40,
    text_candidates: int = 40,
    vector_weight: float = 0.72,
    text_weight: float = 0.28,
) -> list[dict[str, Any]]:
    embedding_sql = vector_literal(embedding)
    rows = conn.execute(
        """
        WITH query_input AS (
            SELECT
                %s::vector AS query_embedding,
                websearch_to_tsquery('russian', %s) AS query_terms
        ),
        vector_results AS (
            SELECT
                c.chunk_id,
                row_number() OVER (ORDER BY c.embedding <=> q.query_embedding) AS vector_rank,
                1 - (c.embedding <=> q.query_embedding) AS vector_score
            FROM doc_chunks c
            CROSS JOIN query_input q
            ORDER BY c.embedding <=> q.query_embedding
            LIMIT %s
        ),
        text_results AS (
            SELECT
                c.chunk_id,
                row_number() OVER (ORDER BY ts_rank_cd(c.search_tsv, q.query_terms) DESC) AS text_rank,
                ts_rank_cd(c.search_tsv, q.query_terms) AS text_score
            FROM doc_chunks c
            CROSS JOIN query_input q
            WHERE c.search_tsv @@ q.query_terms
            ORDER BY ts_rank_cd(c.search_tsv, q.query_terms) DESC
            LIMIT %s
        ),
        combined AS (
            SELECT chunk_id FROM vector_results
            UNION
            SELECT chunk_id FROM text_results
        )
        SELECT
            c.chunk_id,
            c.doc_id,
            c.source_name,
            c.index_code,
            c.document_title,
            c.version,
            c.chunk_type,
            c.citation_label,
            c.raw_text,
            c.section_path,
            c.section_labels,
            c.section_title,
            c.subsection_title,
            c.item_number,
            c.heading_number,
            c.appendix_number,
            c.appendix_title,
            coalesce(v.vector_rank, 999999) AS vector_rank,
            coalesce(t.text_rank, 999999) AS text_rank,
            coalesce(v.vector_score, 0) AS vector_score,
            coalesce(t.text_score, 0) AS text_score,
            (
                %s * coalesce(1.0 / (60 + v.vector_rank), 0) +
                %s * coalesce(1.0 / (60 + t.text_rank), 0)
            ) AS hybrid_score
        FROM combined ids
        JOIN doc_chunks c ON c.chunk_id = ids.chunk_id
        LEFT JOIN vector_results v ON v.chunk_id = c.chunk_id
        LEFT JOIN text_results t ON t.chunk_id = c.chunk_id
        ORDER BY hybrid_score DESC, vector_score DESC, text_score DESC
        LIMIT %s
        """,
        (
            embedding_sql,
            query,
            vector_candidates,
            text_candidates,
            vector_weight,
            text_weight,
            limit,
        ),
    ).fetchall()
    return list(rows)
