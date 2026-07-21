from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from fastapi.responses import JSONResponse

from document_search.openai_compatible import _stream_completion, health


class OpenAICompatibleTest(unittest.TestCase):
    def test_streams_long_content_in_multiple_chunks(self) -> None:
        content = "а" * 4500
        events = list(
            _stream_completion(
                response_id="chatcmpl-test",
                created=1,
                model="document-search-rag",
                content=content,
            )
        )

        payloads = [
            json.loads(event.removeprefix("data: ").strip())
            for event in events
            if event != "data: [DONE]\n\n"
        ]
        streamed_text = "".join(
            payload["choices"][0]["delta"].get("content", "")
            for payload in payloads
        )

        self.assertEqual(streamed_text, content)
        self.assertGreater(len(payloads), 2)

    def test_health_reports_ready_index_and_profile(self) -> None:
        connection = MagicMock()
        connection_context = MagicMock()
        connection_context.__enter__.return_value = connection
        embedder = MagicMock(index_id="BAAI/bge-m3")
        embedder.embedding_dimension.return_value = 1024
        with (
            patch("document_search.openai_compatible.make_embedder", return_value=embedder),
            patch("document_search.openai_compatible.connect", return_value=connection_context),
            patch("document_search.openai_compatible.acquire_corpus_read_lock") as read_lock,
            patch(
                "document_search.openai_compatible.count_rows",
                return_value={"documents": 3, "chunks": 42},
            ),
            patch(
                "document_search.openai_compatible.validate_embedding_profile",
                return_value={"model_id": "BAAI/bge-m3", "dimension": 1024, "chunks": 42},
            ) as validate_profile,
            patch.dict("os.environ", {"RAG_EMBEDDING_DIM": "1024"}),
        ):
            response = health()

        self.assertIsInstance(response, dict)
        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["index"], {"documents": 3, "chunks": 42})
        read_lock.assert_called_once_with(connection)
        validate_profile.assert_called_once_with(
            connection,
            expected_model="BAAI/bge-m3",
            expected_dimension=1024,
        )

    def test_health_returns_503_when_embedding_model_cannot_load(self) -> None:
        embedder = MagicMock(index_id="BAAI/bge-m3")
        embedder.embedding_dimension.side_effect = RuntimeError("weights are unavailable")
        with (
            patch("document_search.openai_compatible.make_embedder", return_value=embedder),
            patch("document_search.openai_compatible.connect") as connect,
        ):
            response = health()

        self.assertIsInstance(response, JSONResponse)
        self.assertEqual(response.status_code, 503)
        connect.assert_not_called()

    def test_health_returns_503_when_database_is_unavailable(self) -> None:
        with (
            patch(
                "document_search.openai_compatible.make_embedder",
                return_value=MagicMock(
                    index_id="model",
                    embedding_dimension=MagicMock(return_value=1024),
                ),
            ),
            patch("document_search.openai_compatible.connect", side_effect=RuntimeError("offline")),
        ):
            response = health()

        self.assertIsInstance(response, JSONResponse)
        self.assertEqual(response.status_code, 503)
        payload = json.loads(response.body)
        self.assertEqual(payload["status"], "error")
        self.assertIsNone(payload["index"])


if __name__ == "__main__":
    unittest.main()
