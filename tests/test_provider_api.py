from __future__ import annotations

import unittest
from unittest.mock import patch

from document_search.local_embedder import LocalEmbedder
from document_search.provider_api import ProviderChat, clear_embedder_cache, make_embedder


class ProviderChatTest(unittest.TestCase):
    def tearDown(self) -> None:
        clear_embedder_cache()

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

    def test_local_embedder_is_reused_within_process(self) -> None:
        clear_embedder_cache()
        with patch.dict(
            "os.environ",
            {
                "EMBEDDING_PROVIDER": "local",
                "LOCAL_EMBED_ENGINE": "sentence-transformers",
                "LOCAL_EMBED_MODEL": "/opt/models/bge-m3",
                "LOCAL_EMBED_INDEX_ID": "BAAI/bge-m3:test",
                "LOCAL_EMBED_MAX_CONCURRENCY": "1",
            },
            clear=False,
        ):
            first = make_embedder()
            second = make_embedder()

        self.assertIs(first, second)
        self.assertEqual(
            first.index_id,
            "BAAI/bge-m3:test|engine=sentence-transformers|normalize=true",
        )

    def test_local_embedding_dimension_is_computed_only_once(self) -> None:
        embedder = LocalEmbedder(
            model="/opt/models/bge-m3",
            engine="sentence-transformers",
        )
        with patch.object(embedder, "embed_text", return_value=[0.1, 0.2, 0.3]) as embed:
            self.assertEqual(embedder.embedding_dimension(), 3)
            self.assertEqual(embedder.embedding_dimension(), 3)

        embed.assert_called_once_with("dimension check")

    def test_local_model_override_cannot_reuse_configured_index_id(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "EMBEDDING_PROVIDER": "local",
                "LOCAL_EMBED_ENGINE": "sentence-transformers",
                "LOCAL_EMBED_MODEL": "/opt/models/bge-m3",
                "LOCAL_EMBED_INDEX_ID": "BAAI/bge-m3:v1",
            },
            clear=False,
        ):
            embedder = make_embedder(model="/tmp/different-model")

        self.assertEqual(
            embedder.index_id,
            "local:/tmp/different-model|engine=sentence-transformers|normalize=true",
        )


if __name__ == "__main__":
    unittest.main()
