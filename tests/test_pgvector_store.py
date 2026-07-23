from __future__ import annotations

import unittest
from unittest.mock import patch

from document_search.pgvector_store import (
    CORPUS_PROMOTION_ADVISORY_LOCK_KEY,
    acquire_corpus_read_lock,
    init_schema,
    load_item_subtree,
    load_structural_chunks,
    parse_structural_reference,
    upsert_chunk,
    validate_embedding_profile,
)
from document_search.retriever import hybrid_search


class FakeResult:
    def __init__(self, rows: list[dict] | None = None) -> None:
        self.rows = rows or []

    def fetchall(self) -> list[dict]:
        return self.rows

    def fetchone(self) -> dict:
        return self.rows[0]


class RecordingConnection:
    def __init__(self, rows: list[dict] | None = None) -> None:
        self.rows = rows or []
        self.calls: list[tuple[str, object | None]] = []
        self.commits = 0
        self.rollbacks = 0

    def execute(self, query: str, parameters: object | None = None) -> FakeResult:
        self.calls.append((query, parameters))
        return FakeResult(self.rows)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


class SequencedConnection(RecordingConnection):
    def __init__(self, responses: list[list[dict]]) -> None:
        super().__init__()
        self.responses = list(responses)

    def execute(self, query: str, parameters: object | None = None) -> FakeResult:
        self.calls.append((query, parameters))
        rows = self.responses.pop(0) if self.responses else []
        return FakeResult(rows)


class PgvectorSchemaTest(unittest.TestCase):
    def test_corpus_read_lock_uses_shared_transaction_advisory_lock(self) -> None:
        conn = RecordingConnection()

        acquire_corpus_read_lock(conn)

        self.assertEqual(
            conn.calls,
            [
                (
                    "SELECT pg_advisory_xact_lock_shared(%s::bigint)",
                    (CORPUS_PROMOTION_ADVISORY_LOCK_KEY,),
                )
            ],
        )

    def test_schema_upgrade_adds_structural_columns_and_indexes(self) -> None:
        conn = RecordingConnection()

        warnings = init_schema(conn, embedding_dim=1024)

        self.assertEqual(warnings, [])
        sql = "\n".join(query for query, _ in conn.calls)
        self.assertIn("section_labels text[]", sql)
        self.assertIn("heading_number text", sql)
        self.assertIn("ADD COLUMN IF NOT EXISTS section_labels", sql)
        self.assertIn("ADD COLUMN IF NOT EXISTS heading_number", sql)
        self.assertIn("doc_chunks_item_number_idx", sql)
        self.assertIn("doc_chunks_section_path_idx", sql)
        self.assertIn("doc_chunks_appendix_number_idx", sql)
        self.assertIn("USING gin(section_path)", sql)

    def test_chunk_upsert_persists_structural_labels(self) -> None:
        conn = RecordingConnection()
        chunk = {
            "chunk_id": "doc::chunk-0001",
            "doc_id": "doc",
            "source_name": "doc.docx",
            "chunk_type": "heading",
            "citation_label": "Документ, раздел 1",
            "raw_text": "1 Введение",
            "searchable_text": "Раздел: 1 Введение\n\n1 Введение",
            "block_ids": ["paragraph-1"],
            "section_path": ["1"],
            "section_labels": ["1 Введение"],
            "heading_number": "1",
            "char_count": 10,
        }

        with patch(
            "document_search.pgvector_store._load_psycopg",
            return_value=(None, None, lambda value: value),
        ):
            upsert_chunk(
                conn,
                chunk,
                embedding=[0.1, 0.2],
                embedding_model="BAAI/bge-m3",
            )

        query, parameters = conn.calls[0]
        self.assertIn("section_labels", query)
        self.assertIn("heading_number", query)
        self.assertEqual(parameters[12], ["1 Введение"])
        self.assertEqual(parameters[16], "1")


class StructuralLookupTest(unittest.TestCase):
    def test_bare_lookup_prefers_exact_item_within_document(self) -> None:
        expected = [{"chunk_id": "doc::chunk-0002", "raw_text": "Текст пункта"}]
        conn = RecordingConnection(expected)

        rows = load_structural_chunks(conn, " 2.1 ", doc_id="doc")

        self.assertEqual(rows, expected)
        query, parameters = conn.calls[0]
        self.assertIn("chunk_type <> 'metadata'", query)
        self.assertIn("item_number = %s", query)
        self.assertIn("'item' AS structural_match", query)
        self.assertNotIn("heading_number = %s OR", query)
        self.assertIn("AND doc_id = %s", query)
        self.assertIn("ORDER BY doc_id, chunk_id", query)
        self.assertEqual(parameters, ("2.1", "doc"))

    def test_empty_reference_does_not_query_database(self) -> None:
        conn = RecordingConnection()

        self.assertEqual(load_structural_chunks(conn, "  "), [])
        self.assertEqual(conn.calls, [])

    def test_appendix_reference_uses_exact_appendix_number(self) -> None:
        conn = RecordingConnection()

        load_structural_chunks(conn, "Приложение № 4", doc_id="policy")

        query, parameters = conn.calls[0]
        self.assertIn("appendix_number = %s", query)
        self.assertEqual(parameters, ("4", "policy"))

    def test_prefixed_item_reference_is_normalized(self) -> None:
        conn = RecordingConnection()

        load_structural_chunks(conn, "пункт 2.3")

        query, parameters = conn.calls[0]
        self.assertIn("item_number = %s", query)
        self.assertNotIn("section_path @>", query)
        self.assertEqual(parameters, ("2.3",))

    def test_prefixed_item_falls_back_to_same_numbered_section(self) -> None:
        expected = [
            {
                "chunk_id": "doc::chunk-0025",
                "raw_text": "2.5.5 Требования к резервному копированию",
            }
        ]
        conn = SequencedConnection([[], expected])

        rows = load_structural_chunks(conn, "пункт 2.5.5", doc_id="policy")

        self.assertEqual(rows, expected)
        self.assertEqual(len(conn.calls), 2)
        first_query, first_parameters = conn.calls[0]
        second_query, second_parameters = conn.calls[1]
        self.assertIn("item_number = %s", first_query)
        self.assertEqual(first_parameters, ("2.5.5", "policy"))
        self.assertIn("heading_number = %s", second_query)
        self.assertIn("section_path @> ARRAY[%s]::text[]", second_query)
        self.assertIn("'section' AS structural_match", second_query)
        self.assertEqual(second_parameters, ("2.5.5", "2.5.5", "policy"))

    def test_prefixed_section_reference_loads_only_section_subtree(self) -> None:
        conn = RecordingConnection()

        load_structural_chunks(conn, "раздел 2.3")

        query, parameters = conn.calls[0]
        self.assertNotIn("item_number = %s", query)
        self.assertIn("heading_number = %s", query)
        self.assertIn("section_path @> ARRAY[%s]::text[]", query)
        self.assertIn("'section' AS structural_match", query)
        self.assertEqual(parameters, ("2.3", "2.3"))

    def test_bare_reference_falls_back_to_section_when_item_is_absent(self) -> None:
        expected = [{"chunk_id": "doc::chunk-0003", "raw_text": "Раздел"}]
        conn = SequencedConnection([[], expected])

        rows = load_structural_chunks(conn, "2.3")

        self.assertEqual(rows, expected)
        self.assertEqual(len(conn.calls), 2)
        self.assertIn("item_number = %s", conn.calls[0][0])
        self.assertIn("section_path @>", conn.calls[1][0])

    def test_composite_appendix_item_reference_is_one_exact_anchor(self) -> None:
        conn = RecordingConnection()

        load_structural_chunks(
            conn,
            "приложение № 3, пункт 2",
            doc_id="policy",
        )

        query, parameters = conn.calls[0]
        self.assertIn("item_number = %s", query)
        self.assertIn("appendix_number = %s", query)
        self.assertEqual(parameters, ("2", "3", "policy"))

    def test_composite_reference_parser_accepts_reverse_word_order(self) -> None:
        reference = parse_structural_reference("пункт 2 приложения № 3")

        self.assertEqual(reference.kind, "item")
        self.assertEqual(reference.number, "2")
        self.assertEqual(reference.appendix_number, "3")
        self.assertEqual(reference.canonical, "приложение № 3, пункт 2")

    def test_bracketed_source_number_is_not_a_structural_reference(self) -> None:
        conn = RecordingConnection()

        self.assertIsNone(parse_structural_reference("[2]"))
        self.assertEqual(load_structural_chunks(conn, "[2]"), [])
        self.assertEqual(conn.calls, [])

    def test_explicit_bracketed_item_stays_in_structural_namespace(self) -> None:
        reference = parse_structural_reference("пункт [2]")

        self.assertEqual(reference.kind, "item")
        self.assertEqual(reference.number, "2")

    def test_appendix_preposition_is_not_treated_as_a_letter_number(self) -> None:
        self.assertIsNone(parse_structural_reference("Приложение к политике доступа"))

    def test_item_subtree_includes_parent_and_dot_descendants(self) -> None:
        conn = RecordingConnection()

        load_structural_chunks(
            conn,
            "пункт 2",
            doc_id="policy",
            include_descendants=True,
        )

        query, parameters = conn.calls[0]
        self.assertIn("item_number = %s", query)
        self.assertIn("strpos(item_number, %s || '.') = 1", query)
        self.assertEqual(parameters, ("2", "2", "policy"))

    def test_item_subtree_helper_can_be_scoped_to_appendix(self) -> None:
        conn = RecordingConnection()

        load_item_subtree(
            conn,
            "2",
            doc_id="policy",
            appendix_number="3",
        )

        query, parameters = conn.calls[0]
        self.assertIn("strpos(item_number, %s || '.') = 1", query)
        self.assertIn("appendix_number = %s", query)
        self.assertEqual(parameters, ("2", "2", "3", "policy"))


class EmbeddingProfileTest(unittest.TestCase):
    def test_accepts_single_matching_embedding_profile(self) -> None:
        conn = RecordingConnection(
            [
                {
                    "embedding_model": "BAAI/bge-m3",
                    "embedding_dimension": 1024,
                    "chunk_count": 37,
                }
            ]
        )

        profile = validate_embedding_profile(
            conn,
            expected_model="BAAI/bge-m3",
            expected_dimension=1024,
        )

        self.assertEqual(
            profile,
            {"model_id": "BAAI/bge-m3", "dimension": 1024, "chunks": 37},
        )
        self.assertIn("vector_dims(embedding)", conn.calls[0][0])

    def test_rejects_mixed_model_ids(self) -> None:
        conn = RecordingConnection(
            [
                {
                    "embedding_model": "BAAI/bge-m3",
                    "embedding_dimension": 1024,
                    "chunk_count": 20,
                },
                {
                    "embedding_model": "another-model",
                    "embedding_dimension": 1024,
                    "chunk_count": 1,
                },
            ]
        )

        with self.assertRaisesRegex(RuntimeError, "mixed model IDs"):
            validate_embedding_profile(
                conn,
                expected_model="BAAI/bge-m3",
                expected_dimension=1024,
            )

    def test_rejects_wrong_model_or_dimension(self) -> None:
        model_conn = RecordingConnection(
            [
                {
                    "embedding_model": "old-model",
                    "embedding_dimension": 1024,
                    "chunk_count": 1,
                }
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "Embedding model mismatch"):
            validate_embedding_profile(
                model_conn,
                expected_model="BAAI/bge-m3",
                expected_dimension=1024,
            )

        dimension_conn = RecordingConnection(
            [
                {
                    "embedding_model": "BAAI/bge-m3",
                    "embedding_dimension": 768,
                    "chunk_count": 1,
                }
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "Embedding dimension mismatch"):
            validate_embedding_profile(
                dimension_conn,
                expected_model="BAAI/bge-m3",
                expected_dimension=1024,
            )

    def test_rejects_empty_index(self) -> None:
        conn = RecordingConnection()
        with self.assertRaisesRegex(RuntimeError, "index is empty"):
            validate_embedding_profile(
                conn,
                expected_model="BAAI/bge-m3",
                expected_dimension=1024,
            )


class RetrieverProjectionTest(unittest.TestCase):
    def test_hybrid_search_returns_structural_fields(self) -> None:
        expected = [{"chunk_id": "chunk-1"}]
        conn = RecordingConnection(expected)

        rows = hybrid_search(conn, query="пункт 2", embedding=[0.1, 0.2], limit=3)

        self.assertEqual(rows, expected)
        query, parameters = conn.calls[0]
        self.assertIn("c.section_labels", query)
        self.assertIn("c.heading_number", query)
        self.assertEqual(parameters[-1], 3)


if __name__ == "__main__":
    unittest.main()
