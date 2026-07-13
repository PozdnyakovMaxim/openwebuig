from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from document_search.query_router import RouteDecision
from document_search.rag_service import answer_question
from document_search.service_queries import documents_answer


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


if __name__ == "__main__":
    unittest.main()
