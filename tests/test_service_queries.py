from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from document_search.query_router import RouteDecision
from document_search.rag_service import (
    _distinct_structural_anchors,
    _resolved_structural_requests,
    _select_document,
    answer_question,
)
from document_search.service_queries import (
    document_not_found_answer,
    document_section_ambiguous_answer,
    documents_answer,
    full_document_answer,
)


class ServiceQueriesTest(unittest.TestCase):
    def test_ambiguity_options_include_unique_document_identifiers(self) -> None:
        candidates = [
            {
                "document_title": "Политика",
                "source_name": "policy-v1.docx",
                "index_code": "POL-1",
                "version": "1.0",
            },
            {
                "document_title": "Политика",
                "source_name": "policy-v2.docx",
                "index_code": "POL-2",
                "version": "2.0",
            },
        ]

        not_found = document_not_found_answer("Политика", candidates)
        section = document_section_ambiguous_answer("пункт 2", candidates)

        for answer in (not_found, section):
            self.assertIn("policy-v1.docx", answer)
            self.assertIn("индекс POL-2", answer)
            self.assertIn("версия 2.0", answer)

    def test_duplicate_exact_titles_require_a_unique_source_or_index(self) -> None:
        candidates = [
            {
                "doc_id": "v1",
                "document_title": "Политика резервного копирования",
                "source_name": "backup-v1.docx",
                "index_code": "BACKUP-1",
                "match_score": 1.0,
            },
            {
                "doc_id": "v2",
                "document_title": "Политика резервного копирования",
                "source_name": "backup-v2.docx",
                "index_code": "BACKUP-2",
                "match_score": 1.0,
            },
        ]

        self.assertIsNone(
            _select_document(candidates, "Политика резервного копирования")
        )
        self.assertEqual(
            _select_document(candidates, "backup-v2.docx")["doc_id"],
            "v2",
        )
        self.assertEqual(
            _select_document(
                candidates,
                "Политика резервного копирования (backup-v2.docx; индекс BACKUP-2)",
            )["doc_id"],
            "v2",
        )

    def test_general_route_does_not_use_embeddings_or_database(self) -> None:
        with (
            patch("document_search.rag_service.make_chat", return_value=MagicMock()),
            patch(
                "document_search.rag_service.route_query",
                return_value=RouteDecision(route="general", answer="Всё хорошо, спасибо! Чем могу помочь?"),
            ),
            patch("document_search.rag_service.make_embedder") as make_embedder,
            patch("document_search.rag_service.connect") as connect,
        ):
            answer = answer_question("Как проходит день?", chat_model="test-model")

        self.assertEqual(answer.route, "general")
        self.assertEqual(answer.sources, [])
        self.assertEqual(answer.rows, [])
        self.assertIn("Всё хорошо", answer.answer)
        make_embedder.assert_not_called()
        connect.assert_not_called()

    def test_documents_route_reads_complete_document_registry(self) -> None:
        documents = [
            {
                "document_title": "Политика резервного копирования",
                "source_name": "backup.docx",
            }
        ]
        connection = MagicMock()
        connection_context = MagicMock()
        connection_context.__enter__.return_value = connection
        with (
            patch("document_search.rag_service.make_chat", return_value=MagicMock()),
            patch(
                "document_search.rag_service.route_query",
                return_value=RouteDecision(route="documents"),
            ),
            patch("document_search.rag_service.connect", return_value=connection_context),
            patch(
                "document_search.rag_service.list_documents",
                return_value=(documents, 1),
            ) as list_documents,
        ):
            answer = answer_question("Что у тебя есть в базе?", chat_model="test-model")

        self.assertEqual(answer.route, "documents")
        self.assertIn("Политика резервного копирования", answer.answer)
        list_documents.assert_called_once_with(connection)

    def test_documents_route_filters_catalog_by_topic(self) -> None:
        connection = MagicMock()
        connection_context = MagicMock()
        connection_context.__enter__.return_value = connection
        embedder = MagicMock(index_id="BAAI/bge-m3")
        embedder.embed_text.return_value = [0.1, 0.2]
        rows = [
            {
                "doc_id": "backup",
                "document_title": "Политика резервного копирования",
                "source_name": "backup.docx",
                "vector_score": 0.74,
                "text_score": 0.08,
                "vector_rank": 1,
                "text_rank": 1,
                "hybrid_score": 0.02,
            },
            {
                "doc_id": "backup",
                "document_title": "Политика резервного копирования",
                "source_name": "backup.docx",
                "vector_score": 0.69,
                "text_score": 0.04,
                "vector_rank": 2,
                "text_rank": 2,
                "hybrid_score": 0.018,
            },
            {
                "doc_id": "security",
                "document_title": "Политика информационной безопасности",
                "source_name": "security.docx",
                "vector_score": 0.36,
                "text_score": 0.0,
                "vector_rank": 90,
                "text_rank": 999999,
                "hybrid_score": 0.004,
            },
        ]
        with (
            patch("document_search.rag_service.make_chat", return_value=MagicMock()),
            patch(
                "document_search.rag_service.route_query",
                return_value=RouteDecision(
                    route="documents",
                    retrieval_query="резервное копирование",
                ),
            ),
            patch("document_search.rag_service.make_embedder", return_value=embedder),
            patch("document_search.rag_service.connect", return_value=connection_context),
            patch("document_search.rag_service.validate_embedding_profile") as validate_profile,
            patch("document_search.rag_service.hybrid_search", return_value=rows) as hybrid_search,
            patch("document_search.rag_service.list_documents") as list_documents,
        ):
            answer = answer_question(
                "Какие документы есть про резервное копирование?",
                chat_model="test-model",
            )

        self.assertEqual(answer.route, "documents")
        self.assertIn("по теме «резервное копирование»", answer.answer)
        self.assertEqual(answer.answer.count("Политика резервного копирования"), 1)
        self.assertNotIn("Политика информационной безопасности", answer.answer)
        list_documents.assert_not_called()
        validate_profile.assert_called_once_with(
            connection,
            expected_model="BAAI/bge-m3",
            expected_dimension=2,
        )
        self.assertEqual(hybrid_search.call_args.kwargs["query"], "резервное копирование")

    def test_document_answer_uses_real_metadata(self) -> None:
        answer = documents_answer(
            [
                {
                    "document_title": "Политика резервного копирования",
                    "source_name": "backup.docx",
                }
            ],
            total=1,
        )
        self.assertIn("Сейчас в индексе 1 документ", answer)
        self.assertIn("Политика резервного копирования (backup.docx)", answer)

    def test_full_document_route_returns_all_chunks_in_order(self) -> None:
        connection = MagicMock()
        connection_context = MagicMock()
        connection_context.__enter__.return_value = connection
        document = {
            "doc_id": "backup",
            "document_title": "Политика резервного копирования",
            "source_name": "backup.docx",
            "match_score": 0.91,
        }
        chunks = [
            {"raw_text": "Первый раздел."},
            {"raw_text": "Второй раздел."},
        ]
        with (
            patch("document_search.rag_service.make_chat", return_value=MagicMock()),
            patch(
                "document_search.rag_service.route_query",
                return_value=RouteDecision(
                    route="full_document",
                    document_query="Политика резервного копирования",
                ),
            ),
            patch("document_search.rag_service.connect", return_value=connection_context),
            patch(
                "document_search.rag_service.find_documents",
                return_value=[document],
            ),
            patch(
                "document_search.rag_service.load_document_chunks",
                return_value=chunks,
            ),
            patch(
                "document_search.rag_service.load_source_document_text",
                return_value="Исходный первый раздел.\n\nИсходный второй раздел.",
            ),
            patch("document_search.rag_service.make_embedder") as make_embedder,
        ):
            answer = answer_question(
                "Выведи полный текст политики резервного копирования",
                chat_model="test-model",
            )

        self.assertEqual(answer.route, "full_document")
        self.assertEqual(answer.mode, "service")
        self.assertIn("Исходный первый раздел.", answer.answer)
        self.assertIn("Исходный второй раздел.", answer.answer)
        self.assertNotIn("\n\nПервый раздел.", answer.answer)
        make_embedder.assert_not_called()

    def test_ambiguous_full_document_request_asks_for_clarification(self) -> None:
        with (
            patch("document_search.rag_service.make_chat", return_value=MagicMock()),
            patch(
                "document_search.rag_service.route_query",
                return_value=RouteDecision(
                    route="full_document",
                    answer="Уточните, какой документ нужно вывести.",
                ),
            ),
            patch("document_search.rag_service.connect") as connect,
        ):
            answer = answer_question(
                "Выведи полный текст этого документа",
                chat_model="test-model",
            )

        self.assertEqual(answer.route, "full_document")
        self.assertIn("Уточните", answer.answer)
        connect.assert_not_called()

    def test_full_document_answer_joins_all_chunks(self) -> None:
        answer = full_document_answer(
            {"document_title": "Инструкция"},
            [{"raw_text": "Часть 1"}, {"raw_text": "Часть 2"}],
        )

        self.assertEqual(answer, "Полный текст документа «Инструкция»:\n\nЧасть 1\n\nЧасть 2")

    def test_full_document_answer_prefers_source_text(self) -> None:
        answer = full_document_answer(
            {"document_title": "Инструкция"},
            [{"raw_text": "Текст из чанка"}],
            source_text="Полный исходный текст",
        )

        self.assertEqual(answer, "Полный текст документа «Инструкция»:\n\nПолный исходный текст")

    def test_exact_document_section_uses_structural_lookup_without_embedding(self) -> None:
        connection = MagicMock()
        connection_context = MagicMock()
        connection_context.__enter__.return_value = connection
        document = {
            "doc_id": "backup",
            "document_title": "Политика резервного копирования",
            "source_name": "backup.docx",
            "match_score": 0.95,
        }
        rows = [
            {
                **document,
                "raw_text": "2.3 Резервные копии проверяются ежемесячно.",
                "citation_label": "Политика резервного копирования, пункт 2.3",
            }
        ]
        with (
            patch("document_search.rag_service.make_chat", return_value=MagicMock()),
            patch(
                "document_search.rag_service.route_query",
                return_value=RouteDecision(
                    route="document_section",
                    document_query="Политика резервного копирования",
                    section_query="2.3",
                ),
            ),
            patch("document_search.rag_service.connect", return_value=connection_context),
            patch("document_search.rag_service.find_documents", return_value=[document]),
            patch(
                "document_search.rag_service.load_structural_chunks",
                return_value=rows,
            ) as load_structural_chunks,
            patch("document_search.rag_service.make_embedder") as make_embedder,
        ):
            answer = answer_question(
                "Покажи пункт 2.3 политики резервного копирования",
                chat_model="test-model",
            )

        self.assertEqual(answer.route, "document_section")
        self.assertIn("2.3 Резервные копии", answer.answer)
        self.assertEqual(answer.sources, ["Политика резервного копирования, пункт 2.3"])
        load_structural_chunks.assert_called_once_with(
            connection,
            "2.3",
            doc_id="backup",
            include_descendants=False,
        )
        make_embedder.assert_not_called()

    def test_global_exact_section_asks_to_disambiguate_multiple_documents(self) -> None:
        connection = MagicMock()
        connection_context = MagicMock()
        connection_context.__enter__.return_value = connection
        rows = [
            {"doc_id": "one", "document_title": "Документ один", "raw_text": "2.3 Текст"},
            {"doc_id": "two", "document_title": "Документ два", "raw_text": "2.3 Текст"},
        ]
        with (
            patch("document_search.rag_service.make_chat", return_value=MagicMock()),
            patch(
                "document_search.rag_service.route_query",
                return_value=RouteDecision(route="document_section", section_query="2.3"),
            ),
            patch("document_search.rag_service.connect", return_value=connection_context),
            patch("document_search.rag_service.load_structural_chunks", return_value=rows),
            patch("document_search.rag_service.make_embedder") as make_embedder,
        ):
            answer = answer_question("Покажи пункт 2.3", chat_model="test-model")

        self.assertIn("нескольких местах", answer.answer)
        self.assertIn("Документ один", answer.answer)
        self.assertIn("Документ два", answer.answer)
        make_embedder.assert_not_called()

    def test_same_item_number_in_multiple_appendices_is_ambiguous(self) -> None:
        connection = MagicMock()
        connection_context = MagicMock()
        connection_context.__enter__.return_value = connection
        document = {
            "doc_id": "backup",
            "document_title": "Политика резервного копирования",
            "source_name": "backup.docx",
            "match_score": 1.0,
        }
        rows = [
            {
                **document,
                "chunk_id": "backup::app1-item1",
                "block_ids": ["appendix-1-item-1"],
                "raw_text": "1 Первый вариант",
                "item_number": "1",
                "appendix_number": "1",
                "structural_match": "item",
            },
            {
                **document,
                "chunk_id": "backup::app2-item1",
                "block_ids": ["appendix-2-item-1"],
                "raw_text": "1 Второй вариант",
                "item_number": "1",
                "appendix_number": "2",
                "structural_match": "item",
            },
        ]
        with (
            patch("document_search.rag_service.make_chat", return_value=MagicMock()),
            patch(
                "document_search.rag_service.route_query",
                return_value=RouteDecision(
                    route="document_section",
                    document_query="Политика резервного копирования",
                    section_query="пункт 1",
                ),
            ),
            patch("document_search.rag_service.connect", return_value=connection_context),
            patch("document_search.rag_service.find_documents", return_value=[document]),
            patch("document_search.rag_service.load_structural_chunks", return_value=rows),
        ):
            answer = answer_question(
                "Покажи пункт 1 политики резервного копирования",
                chat_model="test-model",
            )

        self.assertIn("нескольких местах", answer.answer)
        self.assertIn("приложение № 1", answer.answer)
        self.assertIn("приложение № 2", answer.answer)
        self.assertEqual(answer.rows, [])

    def test_item_with_descendants_is_returned_as_one_structural_subtree(self) -> None:
        connection = MagicMock()
        connection_context = MagicMock()
        connection_context.__enter__.return_value = connection
        rows = [
            {
                "doc_id": "backup",
                "document_title": "Политика резервного копирования",
                "chunk_id": "backup::2",
                "block_ids": ["item-2"],
                "raw_text": "2 Основное требование",
                "item_number": "2",
                "structural_match": "item",
                "citation_label": "Политика, пункт 2",
            },
            {
                "doc_id": "backup",
                "document_title": "Политика резервного копирования",
                "chunk_id": "backup::2.1",
                "block_ids": ["item-2.1"],
                "raw_text": "2.1 Подпункт требования",
                "item_number": "2.1",
                "structural_match": "item",
                "citation_label": "Политика, пункт 2.1",
            },
        ]
        with (
            patch("document_search.rag_service.make_chat", return_value=MagicMock()),
            patch(
                "document_search.rag_service.route_query",
                return_value=RouteDecision(
                    route="document_section",
                    section_query="пункт 2",
                    include_descendants=True,
                ),
            ),
            patch("document_search.rag_service.connect", return_value=connection_context),
            patch(
                "document_search.rag_service.load_structural_chunks",
                return_value=rows,
            ) as load_structural_chunks,
        ):
            answer = answer_question(
                "Покажи пункт 2 вместе с подпунктами",
                chat_model="test-model",
            )

        self.assertIn("2 Основное требование", answer.answer)
        self.assertIn("2.1 Подпункт требования", answer.answer)
        self.assertNotIn("нескольких местах", answer.answer)
        load_structural_chunks.assert_called_once_with(
            connection,
            "пункт 2",
            include_descendants=True,
        )

    def test_both_structural_options_are_loaded_and_combined(self) -> None:
        connection = MagicMock()
        connection_context = MagicMock()
        connection_context.__enter__.return_value = connection
        document = {
            "doc_id": "backup",
            "document_title": "Политика",
            "source_name": "backup.docx",
            "match_score": 1.0,
        }
        selected_rows = [
            [
                {
                    **document,
                    "chunk_id": "backup::app2-item2",
                    "raw_text": "Приложение 2, пункт 2",
                    "item_number": "2",
                    "appendix_number": "2",
                    "structural_match": "item",
                    "citation_label": "Политика, приложение 2, пункт 2",
                }
            ],
            [
                {
                    **document,
                    "chunk_id": "backup::app3-item2",
                    "raw_text": "Приложение 3, пункт 2",
                    "item_number": "2",
                    "appendix_number": "3",
                    "structural_match": "item",
                    "citation_label": "Политика, приложение 3, пункт 2",
                }
            ],
        ]
        resolved_reference = {
            "reference_type": "list_selection",
            "selection": "all",
            "anchors": [
                {
                    "document_query": "Политика (backup.docx)",
                    "section_query": "приложение № 2, пункт 2",
                },
                {
                    "document_query": "Политика (backup.docx)",
                    "section_query": "приложение № 3, пункт 2",
                },
            ],
        }
        with (
            patch("document_search.rag_service.make_chat", return_value=MagicMock()),
            patch(
                "document_search.rag_service.route_query",
                return_value=RouteDecision(
                    route="rag",
                    resolved_reference=resolved_reference,
                ),
            ),
            patch("document_search.rag_service.connect", return_value=connection_context),
            patch("document_search.rag_service.find_documents", return_value=[document]),
            patch(
                "document_search.rag_service.load_structural_chunks",
                side_effect=selected_rows,
            ) as load_structural_chunks,
            patch("document_search.rag_service.make_embedder") as make_embedder,
        ):
            answer = answer_question("оба", chat_model="test-model")

        self.assertEqual(answer.route, "document_section")
        self.assertIn("Приложение 2, пункт 2", answer.answer)
        self.assertIn("Приложение 3, пункт 2", answer.answer)
        self.assertEqual(load_structural_chunks.call_count, 2)
        self.assertEqual(len(answer.rows), 2)
        make_embedder.assert_not_called()

    def test_implicit_reference_to_multiple_citations_asks_for_source(self) -> None:
        resolved_reference = {
            "reference_type": "citation_ambiguity",
            "items": [
                {"source_number": 1, "text": "Политика, пункт 3.1"},
                {"source_number": 2, "text": "Политика, пункт 4.2"},
            ],
        }
        with (
            patch("document_search.rag_service.make_chat", return_value=MagicMock()),
            patch(
                "document_search.rag_service.route_query",
                return_value=RouteDecision(
                    route="rag",
                    resolved_reference=resolved_reference,
                ),
            ),
            patch("document_search.rag_service.connect") as connect,
        ):
            answer = answer_question(
                "Выведи подпункт, на который ты сослался",
                chat_model="test-model",
            )

        self.assertEqual(answer.route, "document_section")
        self.assertIn("Источник [1]", answer.answer)
        self.assertIn("Источник [2]", answer.answer)
        connect.assert_not_called()

    def test_citation_document_does_not_replace_explicit_requested_item(self) -> None:
        decision = RouteDecision(
            route="document_section",
            section_query="пункт 3",
            resolved_reference={
                "reference_type": "citation",
                "anchors": [
                    {
                        "document_query": "Политика доступа",
                        "section_query": "пункт 4.2",
                    }
                ],
            },
        )

        self.assertEqual(
            _resolved_structural_requests(decision),
            [
                {
                    "document_query": "Политика доступа",
                    "section_query": "пункт 3",
                }
            ],
        )

    def test_ambiguous_both_returns_clarification_without_database(self) -> None:
        resolved_reference = {
            "reference_type": "list_selection_ambiguity",
            "available_count": 3,
        }
        with (
            patch("document_search.rag_service.make_chat", return_value=MagicMock()),
            patch(
                "document_search.rag_service.route_query",
                return_value=RouteDecision(
                    route="rag",
                    resolved_reference=resolved_reference,
                ),
            ),
            patch("document_search.rag_service.connect") as connect,
        ):
            answer = answer_question("оба", chat_model="test-model")

        self.assertEqual(answer.route, "document_section")
        self.assertIn("3 вариантов", answer.answer)
        connect.assert_not_called()

    def test_same_item_subtrees_in_different_sections_remain_ambiguous(self) -> None:
        rows = [
            {
                "doc_id": "policy",
                "item_number": "2",
                "section_path": ["1"],
                "structural_match": "item",
            },
            {
                "doc_id": "policy",
                "item_number": "2",
                "section_path": ["4"],
                "structural_match": "item",
            },
        ]

        anchors = _distinct_structural_anchors(
            rows,
            "пункт 2",
            include_descendants=True,
        )

        self.assertEqual(len(anchors), 2)


if __name__ == "__main__":
    unittest.main()
