from __future__ import annotations

import unittest

from document_search.answering import build_messages
from document_search.chat_history import build_retrieval_query, normalize_history


class ChatHistoryTest(unittest.TestCase):
    def test_normalize_history_excludes_current_user_message(self) -> None:
        messages = [
            {"role": "user", "content": "Какие правила резервного копирования?"},
            {"role": "assistant", "content": "Резервное копирование описано в политике."},
            {"role": "user", "content": "А кто отвечает за контроль?"},
        ]

        history = normalize_history(messages, max_messages=6)

        self.assertEqual(
            history,
            [
                {"role": "user", "content": "Какие правила резервного копирования?"},
                {"role": "assistant", "content": "Резервное копирование описано в политике."},
            ],
        )

    def test_normalize_history_keeps_recent_messages(self) -> None:
        messages = [
            {"role": "user", "content": "Первый вопрос"},
            {"role": "assistant", "content": "Первый ответ"},
            {"role": "user", "content": "Второй вопрос"},
            {"role": "assistant", "content": "Второй ответ"},
            {"role": "user", "content": "Текущий вопрос"},
        ]

        history = normalize_history(messages, max_messages=2)

        self.assertEqual(
            history,
            [
                {"role": "user", "content": "Второй вопрос"},
                {"role": "assistant", "content": "Второй ответ"},
            ],
        )

    def test_build_messages_includes_chat_history(self) -> None:
        messages = build_messages(
            "А кто отвечает за контроль?",
            [{"citation_label": "Документ.docx", "raw_text": "Ответственное подразделение выполняет контроль."}],
            chat_history=[
                {"role": "user", "content": "Какие правила резервного копирования?"},
                {"role": "assistant", "content": "Резервное копирование описано в политике."},
            ],
        )

        prompt = messages[-1]["content"]

        self.assertIn("История диалога:", prompt)
        self.assertIn("Пользователь: Какие правила резервного копирования?", prompt)
        self.assertIn("Ассистент: Резервное копирование описано в политике.", prompt)
        self.assertIn("Текущий вопрос: А кто отвечает за контроль?", prompt)

    def test_build_retrieval_query_uses_recent_user_history(self) -> None:
        query = build_retrieval_query(
            "А кто отвечает за контроль?",
            [
                {"role": "user", "content": "Какие правила резервного копирования?"},
                {"role": "assistant", "content": "Резервное копирование описано в политике."},
            ],
        )

        self.assertEqual(query, "Какие правила резервного копирования?\nА кто отвечает за контроль?")


if __name__ == "__main__":
    unittest.main()
