from __future__ import annotations

import unittest
from unittest.mock import patch

from document_search.provider_api import ProviderChat


class ProviderChatTest(unittest.TestCase):
    def test_disables_thinking_for_qwen_requests(self) -> None:
        chat = ProviderChat(
            base_url="https://provider.example/v1",
            api_key="test-key",
            model="Qwen/Qwen3.5-27B-FP8",
        )

        with patch.object(
            chat,
            "_post_json",
            return_value={"choices": [{"message": {"content": "Готово"}}]},
        ) as post_json:
            result = chat.complete([{"role": "user", "content": "Привет"}])

        self.assertEqual(result, "Готово")
        payload = post_json.call_args.args[1]
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertNotIn("max_tokens", payload)


if __name__ == "__main__":
    unittest.main()
