from __future__ import annotations

import json
import unittest

from document_search.query_router import (
    RouteDecision,
    build_router_messages,
    parse_route_decision,
    route_query,
)


class FakeChat:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return self.response


class QueryRouterTest(unittest.TestCase):
    def test_model_decision_routes_general_and_returns_answer(self) -> None:
        chat = FakeChat('{"route":"general","answer":"У меня всё хорошо. Чем помочь?"}')

        decision = route_query(chat, "Как проходит твой день?")

        self.assertEqual(
            decision,
            RouteDecision(route="general", answer="У меня всё хорошо. Чем помочь?"),
        )
        self.assertIsNone(chat.calls[0]["max_tokens"])

    def test_model_decision_routes_document_question_to_rag(self) -> None:
        chat = FakeChat('{"route":"rag","answer":""}')

        decision = route_query(chat, "Кто отвечает за резервное копирование?")

        self.assertEqual(decision, RouteDecision(route="rag"))

    def test_router_receives_recent_conversation_context(self) -> None:
        messages = build_router_messages(
            "А кто это контролирует?",
            chat_history=[
                {"role": "user", "content": "Какие правила резервного копирования?"},
                {"role": "assistant", "content": "Правила описаны в корпоративной политике."},
            ],
        )

        payload = json.loads(messages[1]["content"])

        self.assertEqual(payload["query"], "А кто это контролирует?")
        self.assertEqual(len(payload["history"]), 2)
        self.assertIn("резервного копирования", payload["history"][0]["content"])

    def test_parser_accepts_json_inside_code_fence(self) -> None:
        decision = parse_route_decision(
            '```json\n{"route":"documents","answer":"не используется"}\n```'
        )

        self.assertEqual(decision, RouteDecision(route="documents"))

    def test_invalid_model_response_safely_falls_back_to_rag(self) -> None:
        chat = FakeChat("неструктурированный ответ")

        decision = route_query(chat, "Неоднозначный вопрос")

        self.assertEqual(decision, RouteDecision(route="rag"))


if __name__ == "__main__":
    unittest.main()
