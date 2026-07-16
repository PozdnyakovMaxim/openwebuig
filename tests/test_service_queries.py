from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from document_search.query_router import RouteDecision
from document_search.rag_service import answer_question
from document_search.service_queries import documents_answer, full_document_answer


class ServiceQueriesTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
