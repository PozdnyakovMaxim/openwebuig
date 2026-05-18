from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from document_search.rag_service import answer_question, append_sources
from document_search.settings import load_env_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 5: answer questions with document citations.")
    parser.add_argument("query", help="Question to answer.")
    parser.add_argument("--database-url", default=None, help="PostgreSQL URL. Defaults to DATABASE_URL env.")
    parser.add_argument("--embed-provider", default=None, help="Embedding provider: local or provider.")
    parser.add_argument("--chat-provider", default=None, help="Chat provider: provider.")
    parser.add_argument("--provider-api-base-url", default=None, help="Compatible provider base URL.")
    parser.add_argument("--provider-api-key", default=None, help="Provider API key. Prefer env PROVIDER_API_KEY.")
    parser.add_argument("--embed-model", default=None, help="Embedding model name.")
    parser.add_argument("--chat-model", default=None, help="Optional chat model for generated answers.")
    parser.add_argument("--extractive", action="store_true", help="Do not call a chat model; print source-based answer.")
    parser.add_argument("--limit", type=int, default=6)
    args = parser.parse_args()
    load_env_file()

    answer = answer_question(
        args.query,
        database_url_override=args.database_url,
        embed_provider=args.embed_provider,
        provider_api_base_url=args.provider_api_base_url,
        provider_api_key=args.provider_api_key,
        embed_model=args.embed_model,
        chat_provider=args.chat_provider,
        chat_model=args.chat_model,
        limit=args.limit,
        extractive=args.extractive,
    )
    print(append_sources(answer))
    if answer.mode == "extractive" and not args.extractive:
        print("\nПодсказка: для LLM-ответа задай --chat-model или *_CHAT_MODEL в .env.")
    return 0 if answer.rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
