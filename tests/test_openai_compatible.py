from __future__ import annotations

import json
import unittest

from document_search.openai_compatible import _stream_completion


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


if __name__ == "__main__":
    unittest.main()
