from __future__ import annotations

import json
from io import BytesIO
import unittest
from unittest.mock import MagicMock, patch

from docx import Document
from fastapi import HTTPException, Response
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from document_search.openai_compatible import (
    ChatCompletionRequest,
    ChatMessage,
    _check_document_loader_auth,
    _stream_completion,
    app,
    chat_completions,
    health,
)


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

    def test_openwebui_file_context_bypasses_pgvector_rag(self) -> None:
        chat = MagicMock()
        chat.complete.return_value = "Срок установлен пунктом 2.3.1 [1]."
        request = ChatCompletionRequest(
            model="document-search-rag",
            messages=[
                ChatMessage(
                    role="user",
                    content=(
                        '<context><source id="1" name="policy.docx">'
                        "2.3.1 Срок хранения составляет 30 дней."
                        "</source></context>\n"
                        "User Query: Какой срок хранения?"
                    ),
                )
            ],
            max_tokens=321,
        )
        response = Response()
        with (
            patch.dict("os.environ", {"OPENAI_COMPAT_API_KEY": ""}),
            patch("document_search.openai_compatible.has_chat_config", return_value=True),
            patch("document_search.openai_compatible.make_chat", return_value=chat),
            patch("document_search.openai_compatible.answer_question") as answer_question,
        ):
            payload = chat_completions(request, response)

        self.assertIsInstance(payload, dict)
        assert isinstance(payload, dict)
        self.assertEqual(payload["rag_metrics"]["route"], "file_context")
        self.assertEqual(
            payload["choices"][0]["message"]["content"],
            "Срок установлен пунктом 2.3.1 [1].",
        )
        answer_question.assert_not_called()
        messages = chat.complete.call_args.args[0]
        self.assertIn("2.3.1 Срок хранения", messages[-1]["content"])
        self.assertEqual(chat.complete.call_args.kwargs["max_tokens"], 321)

    def test_document_loader_auth_uses_compat_key_by_default(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "OPENAI_COMPAT_API_KEY": "shared-secret",
                "OPENWEBUI_DOCUMENT_LOADER_API_KEY": "",
            },
        ):
            _check_document_loader_auth("Bearer shared-secret")

            with self.assertRaises(HTTPException) as raised:
                _check_document_loader_auth("Bearer wrong-secret")
            self.assertEqual(raised.exception.status_code, 401)

    def test_document_loader_auth_fails_closed_without_any_key(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "OPENAI_COMPAT_API_KEY": "",
                "OPENWEBUI_DOCUMENT_LOADER_API_KEY": "",
            },
        ):
            with self.assertRaises(HTTPException) as raised:
                _check_document_loader_auth(None)

        self.assertEqual(raised.exception.status_code, 503)

    def test_process_endpoint_contract_and_error_statuses(self) -> None:
        client = TestClient(app)
        headers = {
            "Authorization": "Bearer loader-secret",
            "Content-Type": "text/plain",
            "X-Filename": "notes.txt",
        }
        with patch.dict(
            "os.environ",
            {
                "OPENAI_COMPAT_API_KEY": "compat-secret",
                "OPENWEBUI_DOCUMENT_LOADER_API_KEY": "loader-secret",
                "OPENWEBUI_DOCUMENT_MAX_BYTES": "64",
            },
        ):
            successful = client.put("/process", content="Пункт 2.3".encode(), headers=headers)
            unauthorized = client.put(
                "/process",
                content=b"text",
                headers={**headers, "Authorization": "Bearer compat-secret"},
            )
            unsupported = client.put(
                "/process",
                content=b"binary",
                headers={**headers, "Content-Type": "application/octet-stream", "X-Filename": "a.xlsx"},
            )
            invalid_docx = client.put(
                "/process",
                content=b"not-a-docx",
                headers={**headers, "Content-Type": "application/octet-stream", "X-Filename": "a.docx"},
            )
            too_large = client.put(
                "/process",
                content=b"x" * 65,
                headers=headers,
            )

        self.assertEqual(successful.status_code, 200)
        self.assertEqual(successful.json()["page_content"], "Пункт 2.3")
        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(unsupported.status_code, 415)
        self.assertEqual(invalid_docx.status_code, 400)
        self.assertEqual(too_large.status_code, 413)

    def test_process_endpoint_fails_closed_for_missing_key_and_bad_limit(self) -> None:
        client = TestClient(app)
        headers = {"Content-Type": "text/plain", "X-Filename": "notes.txt"}
        with patch.dict(
            "os.environ",
            {
                "OPENAI_COMPAT_API_KEY": "",
                "OPENWEBUI_DOCUMENT_LOADER_API_KEY": "",
                "OPENWEBUI_DOCUMENT_MAX_BYTES": "64",
            },
        ):
            missing_key = client.put("/process", content=b"text", headers=headers)
        with patch.dict(
            "os.environ",
            {
                "OPENAI_COMPAT_API_KEY": "compat-secret",
                "OPENWEBUI_DOCUMENT_LOADER_API_KEY": "",
                "OPENWEBUI_DOCUMENT_MAX_BYTES": "invalid",
            },
        ):
            bad_limit = client.put(
                "/process",
                content=b"text",
                headers={**headers, "Authorization": "Bearer compat-secret"},
            )

        self.assertEqual(missing_key.status_code, 503)
        self.assertEqual(bad_limit.status_code, 503)

    def test_docx_process_output_flows_to_file_context_without_pgvector(self) -> None:
        document = Document()
        document.add_paragraph("Автоматически нумерованное требование", style="List Number")
        buffer = BytesIO()
        document.save(buffer)
        client = TestClient(app)
        loader_headers = {
            "Authorization": "Bearer shared-secret",
            "Content-Type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "X-Filename": "policy.docx",
        }
        with patch.dict(
            "os.environ",
            {
                "OPENAI_COMPAT_API_KEY": "shared-secret",
                "OPENWEBUI_DOCUMENT_LOADER_API_KEY": "",
            },
        ):
            converted = client.put("/process", content=buffer.getvalue(), headers=loader_headers)

        self.assertEqual(converted.status_code, 200)
        extracted_text = converted.json()["page_content"]
        self.assertIn("1. Автоматически нумерованное требование", extracted_text)

        chat = MagicMock()
        chat.complete.return_value = "Требование имеет номер 1 [1]."
        request = ChatCompletionRequest(
            model="document-search-rag",
            messages=[
                ChatMessage(
                    role="user",
                    content=(
                        f'<context><source id="1" name="policy.docx">{extracted_text}</source></context>\n'
                        "User Query: Какой номер требования?"
                    ),
                )
            ],
        )
        with (
            patch.dict("os.environ", {"OPENAI_COMPAT_API_KEY": ""}),
            patch("document_search.openai_compatible.has_chat_config", return_value=True),
            patch("document_search.openai_compatible.make_chat", return_value=chat),
            patch("document_search.openai_compatible.answer_question") as answer_question,
        ):
            payload = chat_completions(request, Response())

        self.assertIsInstance(payload, dict)
        assert isinstance(payload, dict)
        self.assertEqual(payload["rag_metrics"]["route"], "file_context")
        self.assertIn("1. Автоматически нумерованное требование", chat.complete.call_args.args[0][-1]["content"])
        answer_question.assert_not_called()


if __name__ == "__main__":
    unittest.main()
