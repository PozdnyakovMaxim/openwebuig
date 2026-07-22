from __future__ import annotations

import unittest

from document_search.answering import build_messages
from document_search.chat_history import (
    build_retrieval_query,
    normalize_history,
    parse_reference_anchor,
    resolve_citation_reference,
    resolve_dialog_reference,
    resolve_numbered_list_selection,
    resolve_ordinal_reference,
)


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

        self.assertEqual(
            query,
            "Какие правила резервного копирования?\n"
            "Резервное копирование описано в политике.\n"
            "А кто отвечает за контроль?",
        )

    def test_build_retrieval_query_does_not_mix_unrelated_old_topic(self) -> None:
        query = build_retrieval_query(
            "Как оформляется отпуск?",
            [
                {"role": "user", "content": "Какие правила резервного копирования?"},
                {"role": "assistant", "content": "Они описаны в политике ИБ."},
            ],
        )

        self.assertEqual(query, "Как оформляется отпуск?")

    def test_resolves_ordinal_against_latest_assistant_list(self) -> None:
        reference = resolve_ordinal_reference(
            "Раскрой второй пункт подробнее",
            [
                {
                    "role": "assistant",
                    "content": "1. Создать заявку\n2. Согласовать доступ\n3. Закрыть заявку",
                }
            ],
        )

        self.assertEqual(
            reference,
            {
                "position": 2,
                "kind": "пункт",
                "list_label": "2",
                "text": "Согласовать доступ",
            },
        )

    def test_out_of_range_ordinal_does_not_fall_back_to_older_list(self) -> None:
        reference = resolve_ordinal_reference(
            "Покажи четвёртый документ",
            [
                {
                    "role": "assistant",
                    "content": "1. Старый один\n2. Старый два\n3. Старый три\n4. Старый четыре",
                },
                {"role": "user", "content": "Какие документы есть по новой теме?"},
                {
                    "role": "assistant",
                    "content": "1. Новый один\n2. Новый два\n3. Новый три",
                },
            ],
        )

        self.assertIsNone(reference)

    def test_bracketed_source_is_not_reinterpreted_as_document_item(self) -> None:
        history = [
            {
                "role": "assistant",
                "content": (
                    "Требование приведено в документе [2].\n\n"
                    "Источники:\n"
                    "[1] Политика ИБ, пункт 1.1\n"
                    "[2] Политика резервного копирования, пункт 3.4"
                ),
            }
        ]

        reference = resolve_dialog_reference("Покажи полностью источник [2]", history)

        self.assertEqual(reference["reference_type"], "citation")
        self.assertEqual(reference["source_number"], 2)
        self.assertEqual(reference["text"], "Политика резервного копирования, пункт 3.4")
        self.assertEqual(reference["anchors"][0]["section_query"], "пункт 3.4")

    def test_named_source_number_resolves_against_bracketed_source_list(self) -> None:
        reference = resolve_citation_reference(
            "Раскрой источник 2",
            [
                {
                    "role": "assistant",
                    "content": "Источники:\n[1] Первый документ\n[2] Второй документ",
                }
            ],
        )

        self.assertEqual(reference["source_number"], 2)
        self.assertEqual(reference["text"], "Второй документ")

    def test_implicit_singular_citation_is_not_turned_into_item_two(self) -> None:
        reference = resolve_citation_reference(
            "Выведи полностью подпункт, на который ты сослался",
            [
                {
                    "role": "assistant",
                    "content": (
                        "Требования перечислены в источниках.\n\n"
                        "Источники:\n"
                        "[1] Политика, пункт 3.1\n"
                        "[2] Политика, пункт 4.2"
                    ),
                }
            ],
        )

        self.assertEqual(reference["reference_type"], "citation_ambiguity")
        self.assertEqual(reference["source_numbers"], [1, 2])
        self.assertEqual(
            [anchor["section_query"] for anchor in reference["anchors"]],
            ["пункт 3.1", "пункт 4.2"],
        )

    def test_explicit_source_namespace_survives_missing_source_line(self) -> None:
        reference = resolve_citation_reference(
            "Покажи источник [7]",
            [{"role": "assistant", "content": "Источники:\n[1] Документ"}],
        )

        self.assertEqual(reference["reference_type"], "citation")
        self.assertEqual(reference["source_number"], 7)
        self.assertFalse(reference["found"])
        self.assertEqual(reference["anchors"], [])

    def test_explicit_document_item_is_not_a_source_reference(self) -> None:
        history = [
            {
                "role": "assistant",
                "content": "Источники:\n[1] Первый документ\n[2] Второй документ",
            }
        ]

        self.assertIsNone(resolve_citation_reference("Покажи пункт 2", history))
        self.assertIsNone(resolve_citation_reference("Покажи пункт [2]", history))

    def test_both_selects_all_anchors_from_latest_numbered_list(self) -> None:
        reference = resolve_numbered_list_selection(
            "оба",
            [
                {
                    "role": "assistant",
                    "content": (
                        "Ссылка найдена в нескольких местах:\n"
                        "1. Политика (backup.docx) — приложение № 2, пункт 2\n"
                        "2. Политика (backup.docx) — приложение № 3, пункт 2"
                    ),
                }
            ],
        )

        self.assertEqual(reference["selection"], "all")
        self.assertEqual(reference["positions"], [1, 2])
        self.assertEqual(len(reference["anchors"]), 2)
        self.assertEqual(reference["anchors"][0]["appendix_number"], "2")
        self.assertEqual(
            reference["anchors"][1]["section_query"],
            "приложение № 3, пункт 2",
        )

    def test_all_documents_request_is_not_mistaken_for_list_selection(self) -> None:
        history = [{"role": "assistant", "content": "1. Первый\n2. Второй"}]

        self.assertIsNone(resolve_numbered_list_selection("Покажи все документы", history))

    def test_anchor_parser_supports_composite_reference_in_either_order(self) -> None:
        forward = parse_reference_anchor(
            "Политика (backup.docx) — приложение № 3, пункт 2"
        )
        reverse = parse_reference_anchor(
            "Политика (backup.docx) — пункт 2 приложения 3"
        )

        for anchor in (forward, reverse):
            self.assertEqual(anchor["document_query"], "Политика (backup.docx)")
            self.assertEqual(anchor["appendix_number"], "3")
            self.assertEqual(anchor["item_number"], "2")
            self.assertEqual(anchor["section_query"], "приложение № 3, пункт 2")

    def test_structural_bracket_does_not_hide_later_named_source(self) -> None:
        reference = resolve_citation_reference(
            "Покажи пункт [3] из источника [2]",
            [
                {
                    "role": "assistant",
                    "content": "Источники:\n[1] Первая политика, пункт 1\n[2] Вторая политика, пункт 4.2",
                }
            ],
        )

        self.assertIsNotNone(reference)
        self.assertEqual(reference["source_number"], 2)
        self.assertEqual(reference["anchors"][0]["document_query"], "Вторая политика")

    def test_both_is_ambiguous_when_latest_list_has_not_exactly_two_items(self) -> None:
        reference = resolve_numbered_list_selection(
            "оба",
            [{"role": "assistant", "content": "1. Первый\n2. Второй\n3. Третий"}],
        )

        self.assertEqual(reference["reference_type"], "list_selection_ambiguity")
        self.assertEqual(reference["available_count"], 3)

    def test_list_selection_does_not_reuse_an_older_assistant_list(self) -> None:
        reference = resolve_numbered_list_selection(
            "оба",
            [
                {"role": "assistant", "content": "1. Старый первый\n2. Старый второй"},
                {"role": "user", "content": "Расскажи подробнее"},
                {"role": "assistant", "content": "Вот подробное объяснение без списка."},
            ],
        )

        self.assertIsNone(reference)

    def test_explicit_document_tail_keeps_second_item_structural(self) -> None:
        history = [{"role": "assistant", "content": "1. Старый первый\n2. Старый второй"}]

        self.assertIsNone(
            resolve_numbered_list_selection(
                "Покажи второй пункт Политики резервного копирования",
                history,
            )
        )

    def test_reason_for_citations_is_not_an_implicit_source_selection(self) -> None:
        reference = resolve_citation_reference(
            "Почему ты сослался на эти источники?",
            [{"role": "assistant", "content": "Источники:\n[1] Первый\n[2] Второй"}],
        )

        self.assertIsNone(reference)

    def test_source_label_prefix_is_not_part_of_document_query(self) -> None:
        anchor = parse_reference_anchor("Источник [1]: Политика, пункт 3.1")

        self.assertIsNotNone(anchor)
        self.assertEqual(anchor["document_query"], "Политика")

    def test_prose_after_appendix_word_is_not_a_structural_anchor(self) -> None:
        self.assertIsNone(parse_reference_anchor("Приложение к политике доступа"))


if __name__ == "__main__":
    unittest.main()
