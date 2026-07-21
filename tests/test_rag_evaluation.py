from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPTS_DIRECTORY = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIRECTORY))

import evaluate_rag  # noqa: E402


class RagEvaluationTest(unittest.TestCase):
    def test_parse_sources_uses_final_russian_or_english_block(self) -> None:
        russian = (
            "Ответ с цитатой [1].\n\n"
            "Источники:\n"
            "[1] Политика резервного копирования, пункт 2.3\n"
            "[2] Регламент ИБ\n"
            "Служебный хвост"
        )
        english = "Answer.\n\nSources:\n[1] Backup policy.pdf"

        self.assertEqual(
            evaluate_rag.parse_sources(russian),
            ["Политика резервного копирования, пункт 2.3", "Регламент ИБ"],
        )
        self.assertEqual(evaluate_rag.parse_sources(english), ["Backup policy.pdf"])
        self.assertEqual(evaluate_rag.parse_sources("Только ответ [1]."), [])

    def test_parse_api_response_extracts_route_sources_and_server_timing(self) -> None:
        observation = evaluate_rag.parse_api_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": "Ответ [1].\n\nИсточники:\n[1] Политика ИБ"
                        }
                    }
                ],
                "rag_metrics": {
                    "route": "rag",
                    "timings_ms": {"total": 123.456},
                },
            },
            latency_ms=150.239,
        )

        self.assertEqual(observation.route, "rag")
        self.assertEqual(observation.sources, ["Политика ИБ"])
        self.assertEqual(observation.latency_ms, 150.24)
        self.assertEqual(observation.server_total_ms, 123.46)

    def test_validate_config_and_build_messages(self) -> None:
        config = evaluate_rag.validate_config(
            {
                "cases": [
                    {
                        "query": "  Продолжай  ",
                        "history": [
                            {"role": "user", "content": "Первый вопрос"},
                            {"role": "assistant", "content": "Первый ответ"},
                        ],
                        "expected_route": " rag ",
                    }
                ]
            }
        )
        case = config["cases"][0]

        self.assertEqual(case["id"], "case-1")
        self.assertEqual(case["expected_route"], "rag")
        self.assertEqual(
            evaluate_rag.build_messages(case),
            [
                {"role": "user", "content": "Первый вопрос"},
                {"role": "assistant", "content": "Первый ответ"},
                {"role": "user", "content": "Продолжай"},
            ],
        )

    def test_validate_config_rejects_literal_secret_and_bad_expectations(self) -> None:
        with self.assertRaisesRegex(ValueError, "do not store api_key"):
            evaluate_rag.validate_config(
                {"api_key": "secret", "cases": [{"query": "test"}]}
            )
        with self.assertRaisesRegex(ValueError, "expected_source_substrings"):
            evaluate_rag.validate_config(
                {
                    "cases": [
                        {
                            "query": "test",
                            "expected_source_substrings": "source.pdf",
                        }
                    ]
                }
            )
        with self.assertRaisesRegex(ValueError, "max_latency_ms"):
            evaluate_rag.validate_config(
                {"cases": [{"query": "test", "max_latency_ms": 0}]}
            )

    def test_evaluate_case_passes_all_supported_checks_case_insensitively(self) -> None:
        case = evaluate_rag.validate_config(
            {
                "cases": [
                    {
                        "id": "golden",
                        "query": "Кто отвечает?",
                        "expected_route": "RAG",
                        "expected_source_substrings": ["политика резервного"],
                        "expected_title_substrings": ["ПОЛИТИКА РЕЗЕРВНОГО"],
                        "expected_answer_substrings": ["администратор"],
                        "forbidden_substrings": ["ответа нет"],
                        "max_latency_ms": 500,
                    }
                ]
            }
        )["cases"][0]
        observation = evaluate_rag.ApiObservation(
            answer=(
                "Ответственный — Администратор. [1]\n\nИсточники:\n"
                "[1] Политика резервного копирования, пункт 2.3"
            ),
            route="rag",
            sources=["Политика резервного копирования, пункт 2.3"],
            rag_metrics={"route": "rag", "timings_ms": {"total": 100}},
            latency_ms=200,
        )

        result = evaluate_rag.evaluate_case(case, observation)

        self.assertTrue(result["passed"])
        self.assertEqual(result["failures"], [])
        self.assertEqual(result["checks"]["route"], True)
        self.assertEqual(result["checks"]["sources"], {"matched": 1, "expected": 1})
        self.assertEqual(result["checks"]["titles"], {"matched": 1, "expected": 1})
        self.assertEqual(result["checks"]["latency"], True)

    def test_evaluate_case_reports_every_failed_required_check(self) -> None:
        case = evaluate_rag.validate_config(
            {
                "cases": [
                    {
                        "id": "broken",
                        "query": "test",
                        "expected_route": "rag",
                        "expected_source_substrings": ["Expected source"],
                        "expected_title_substrings": ["Expected title"],
                        "expected_answer_substrings": ["Expected answer"],
                        "forbidden_substrings": ["secret"],
                        "max_latency_ms": 10,
                    }
                ]
            }
        )["cases"][0]
        observation = evaluate_rag.ApiObservation(
            answer="This leaked a SECRET.",
            route="general",
            sources=["Other source"],
            rag_metrics={"route": "general"},
            latency_ms=20,
        )

        result = evaluate_rag.evaluate_case(case, observation)

        self.assertFalse(result["passed"])
        self.assertEqual(len(result["failures"]), 6)
        self.assertTrue(any(failure.startswith("route:") for failure in result["failures"]))
        self.assertTrue(any("source substring" in failure for failure in result["failures"]))
        self.assertTrue(any("title substring" in failure for failure in result["failures"]))
        self.assertTrue(any("answer substring" in failure for failure in result["failures"]))
        self.assertTrue(any("forbidden substring" in failure for failure in result["failures"]))
        self.assertTrue(any(failure.startswith("latency:") for failure in result["failures"]))

    def test_summary_calculates_route_recall_and_latency(self) -> None:
        results = [
            self._result(
                passed=True,
                route=True,
                source=(2, 2),
                title=(1, 1),
                latency=100,
            ),
            self._result(
                passed=False,
                route=False,
                source=(1, 2),
                title=(0, 1),
                latency=300,
            ),
            self._result(
                passed=True,
                route=None,
                source=(0, 0),
                title=(0, 0),
                latency=200,
            ),
        ]

        summary = evaluate_rag.summarize_results(results)

        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["passed"], 2)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(
            summary["route_accuracy"], {"correct": 1, "expected": 2, "value": 0.5}
        )
        self.assertEqual(
            summary["source_recall"], {"matched": 3, "expected": 4, "value": 0.75}
        )
        self.assertEqual(
            summary["title_recall"], {"matched": 1, "expected": 2, "value": 0.5}
        )
        self.assertEqual(
            summary["latency_ms"],
            {"count": 3, "mean": 200.0, "p50": 200.0, "p95": 300.0, "max": 300.0},
        )

    def test_main_returns_nonzero_on_failed_case_without_network_or_secret(self) -> None:
        config = {"cases": [{"id": "route", "query": "Кто ты?", "expected_route": "identity"}]}
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "golden.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            observation = evaluate_rag.ApiObservation(
                answer="Ответ",
                route="rag",
                sources=[],
                rag_metrics={"route": "rag"},
                latency_ms=5,
            )
            with (
                patch.object(evaluate_rag, "call_chat_api", return_value=observation) as call_api,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                exit_code = evaluate_rag.main(["--config", str(config_path)])

        self.assertEqual(exit_code, 1)
        self.assertEqual(call_api.call_args.kwargs["api_key"], "")

    @staticmethod
    def _result(
        *,
        passed: bool,
        route: bool | None,
        source: tuple[int, int],
        title: tuple[int, int],
        latency: float,
    ) -> dict[str, object]:
        return {
            "passed": passed,
            "latency_ms": latency,
            "checks": {
                "route": route,
                "sources": {"matched": source[0], "expected": source[1]},
                "titles": {"matched": title[0], "expected": title[1]},
            },
        }


if __name__ == "__main__":
    unittest.main()
