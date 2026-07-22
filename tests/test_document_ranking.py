from __future__ import annotations

import unittest

from document_search.document_ranking import rank_documents


def _row(
    doc_id: str,
    *,
    vector: float,
    text: float = 0.0,
    hybrid: float = 0.0,
    vector_rank: int = 999_999,
    text_rank: int = 999_999,
) -> dict[str, object]:
    return {
        "doc_id": doc_id,
        "source_name": f"{doc_id}.docx",
        "document_title": doc_id.title(),
        "vector_score": vector,
        "text_score": text,
        "hybrid_score": hybrid,
        "vector_rank": vector_rank,
        "text_rank": text_rank,
    }


class DocumentRankingTest(unittest.TestCase):
    def test_aggregates_chunks_and_keeps_best_document_signals(self) -> None:
        rows = [
            _row(
                "backup",
                vector=0.61,
                text=0.08,
                hybrid=0.014,
                vector_rank=3,
                text_rank=2,
            ),
            _row(
                "backup",
                vector=0.72,
                text=0.0,
                hybrid=0.012,
                vector_rank=1,
            ),
            _row(
                "archive",
                vector=0.63,
                text=0.04,
                hybrid=0.011,
                vector_rank=2,
                text_rank=7,
            ),
        ]

        documents = rank_documents(rows)

        self.assertEqual([item["doc_id"] for item in documents], ["backup", "archive"])
        self.assertEqual(documents[0]["matched_chunks"], 2)
        self.assertEqual(documents[0]["best_vector_score"], 0.72)
        self.assertEqual(documents[0]["best_text_score"], 0.08)
        self.assertEqual(documents[0]["best_vector_rank"], 1)
        self.assertEqual(documents[0]["best_text_rank"], 2)

    def test_strong_vector_only_match_is_preserved(self) -> None:
        rows = [
            _row("semantic", vector=0.70, hybrid=0.011, vector_rank=1),
            _row("background", vector=0.47, hybrid=0.010, vector_rank=2),
        ]

        documents = rank_documents(rows)

        self.assertEqual([item["doc_id"] for item in documents], ["semantic"])
        self.assertEqual(documents[0]["text_signal"], 0.0)
        self.assertGreater(documents[0]["vector_signal"], 0.0)

    def test_strong_lexical_only_match_is_preserved(self) -> None:
        rows = [
            _row(
                "lexical",
                vector=0.31,
                text=0.12,
                hybrid=0.009,
                text_rank=1,
            ),
            _row("noise", vector=0.44, hybrid=0.008, vector_rank=1),
        ]

        documents = rank_documents(rows)

        self.assertEqual([item["doc_id"] for item in documents], ["lexical"])
        self.assertEqual(documents[0]["vector_signal"], 0.0)
        self.assertGreater(documents[0]["text_signal"], 0.0)

    def test_flat_low_score_tail_is_removed_after_relevant_documents(self) -> None:
        rows = [
            _row("best", vector=0.74, hybrid=0.016, vector_rank=1),
            _row("also-relevant", vector=0.63, hybrid=0.015, vector_rank=2),
            _row("tail-1", vector=0.49, hybrid=0.014, vector_rank=3),
            _row("tail-2", vector=0.485, hybrid=0.013, vector_rank=4),
            _row("tail-3", vector=0.48, hybrid=0.012, vector_rank=5),
        ]

        documents = rank_documents(rows)

        self.assertEqual(
            [item["doc_id"] for item in documents],
            ["best", "also-relevant"],
        )

    def test_rrf_score_alone_cannot_make_background_candidates_relevant(self) -> None:
        rows = [
            _row("rrf-1", vector=0.40, hybrid=0.0163, vector_rank=1),
            _row("rrf-2", vector=0.39, hybrid=0.0161, vector_rank=2),
        ]

        self.assertEqual(rank_documents(rows), [])

    def test_vector_and_text_channels_have_independent_relative_cutoffs(self) -> None:
        rows = [
            _row(
                "exact-words",
                vector=0.36,
                text=0.15,
                hybrid=0.016,
                text_rank=1,
            ),
            _row("semantic", vector=0.71, hybrid=0.015, vector_rank=1),
            _row("tail", vector=0.48, hybrid=0.014, vector_rank=2),
        ]

        documents = rank_documents(rows)

        self.assertEqual(
            {item["doc_id"] for item in documents},
            {"exact-words", "semantic"},
        )

    def test_limit_and_ties_are_deterministic(self) -> None:
        rows = [
            _row("first", vector=0.70, hybrid=0.01, vector_rank=1),
            _row("second", vector=0.70, hybrid=0.01, vector_rank=1),
            _row("third", vector=0.70, hybrid=0.01, vector_rank=1),
        ]

        documents = rank_documents(rows, limit=2)

        self.assertEqual([item["doc_id"] for item in documents], ["first", "second"])

    def test_invalid_or_unidentified_rows_do_not_create_results(self) -> None:
        rows = [
            {"vector_score": 0.99, "hybrid_score": 0.016},
            {
                "doc_id": "invalid",
                "vector_score": "not-a-number",
                "text_score": None,
                "hybrid_score": float("nan"),
            },
        ]

        self.assertEqual(rank_documents(rows), [])

    def test_cutoffs_are_configurable_for_evaluation(self) -> None:
        rows = [_row("domain-match", vector=0.43, vector_rank=1)]

        self.assertEqual(rank_documents(rows), [])
        documents = rank_documents(
            rows,
            min_vector_score=0.40,
            min_document_score=0.01,
        )
        self.assertEqual([item["doc_id"] for item in documents], ["domain-match"])

    def test_weak_best_channel_does_not_bypass_global_relative_cutoff(self) -> None:
        rows = [
            _row("exact-words", vector=0.30, text=0.15, text_rank=1),
            _row("weak-semantic", vector=0.48, vector_rank=1),
        ]

        documents = rank_documents(rows)

        self.assertEqual([item["doc_id"] for item in documents], ["exact-words"])


if __name__ == "__main__":
    unittest.main()
