from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from document_search.pgvector_store import (
    acquire_corpus_read_lock,
    connect,
    count_rows,
    database_url,
    resolve_embedding_index_id,
    sample_vector_search,
    validate_embedding_profile,
)
from document_search.provider_api import make_embedder


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate pgvector index and run a sample semantic search.")
    parser.add_argument("--database-url", default=None, help="PostgreSQL URL. Defaults to DATABASE_URL env.")
    parser.add_argument("--embed-provider", default=None, help="Embedding provider: local or provider.")
    parser.add_argument("--provider-api-base-url", default=None, help="Compatible provider base URL.")
    parser.add_argument("--provider-api-key", default=None, help="Provider API key. Prefer env PROVIDER_API_KEY.")
    parser.add_argument("--embed-model", default=None, help="Embedding model name.")
    parser.add_argument("--query", default="Что указано в документах?", help="Smoke-test query.")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    embedder = make_embedder(
        provider=args.embed_provider,
        provider_api_base_url=args.provider_api_base_url,
        provider_api_key=args.provider_api_key,
        model=args.embed_model,
    )
    embedding = embedder.embed_text(args.query)

    url = database_url(args.database_url)
    with connect(url) as conn:
        acquire_corpus_read_lock(conn)
        counts = count_rows(conn)
        validate_embedding_profile(
            conn,
            expected_model=resolve_embedding_index_id(embedder),
            expected_dimension=len(embedding),
        )
        rows = sample_vector_search(conn, embedding=embedding, limit=args.limit)

    print(f"Documents: {counts['documents']}")
    print(f"Chunks: {counts['chunks']}")
    print(f"Query: {args.query}")
    for index, row in enumerate(rows, start=1):
        snippet = " ".join(str(row["raw_text"]).split())[:280]
        score = float(row["vector_score"])
        print(f"{index}. score={score:.4f} {row['citation_label']}")
        print(f"   {snippet}")

    if not rows:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
