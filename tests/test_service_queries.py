from __future__ import annotations

import unittest

from document_search.service_queries import classify_service_query, documents_answer


class ServiceQueriesTest(unittest.TestCase):
    def test_identity_query_does_not_require_rag(self) -> None:
        self.assertEqual(classify_service_query("Кто ты?"), "identity")
        self.assertEqual(classify_service_query("Представься"), "identity")

    def test_capabilities_query_does_not_require_rag(self) -> None:
        self.assertEqual(classify_service_query("Что ты умеешь?"), "capabilities")

    def test_document_list_query_does_not_require_rag(self) -> None:
        self.assertEqual(classify_service_query("Перечисли все доступные документы"), "documents")
        self.assertEqual(classify_service_query("С какими документами ты работаешь?"), "documents")

    def test_document_topic_question_still_uses_rag(self) -> None:
        self.assertIsNone(classify_service_query("Какие документы нужны для удаленного доступа?"))
        self.assertIsNone(classify_service_query("Кто отвечает за резервное копирование?"))

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
