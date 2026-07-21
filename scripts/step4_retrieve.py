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
    database_url,
    resolve_embedding_index_id,
    validate_embedding_profile,
)
from document_search.provider_api import make_embedder
from document_search.retriever import hybrid_search


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 4: hybrid retrieval over pgvector chunks.")
    parser.add_argument("query", help="Question or search query.")
    parser.add_argument("--database-url", default=None, help="PostgreSQL URL. Defaults to DATABASE_URL env.")
    parser.add_argument("--embed-provider", default=None, help="Embedding provider: local or provider.")
    parser.add_argument("--provider-api-base-url", default=None, help="Compatible provider base URL.")
    parser.add_argument("--provider-api-key", default=None, help="Provider API key. Prefer env PROVIDER_API_KEY.")
    parser.add_argument("--embed-model", default=None, help="Embedding model name.")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--vector-candidates", type=int, default=40)
    parser.add_argument("--text-candidates", type=int, default=40)
    args = parser.parse_args()

    embedder = make_embedder(
        provider=args.embed_provider,
        provider_api_base_url=args.provider_api_base_url,
        provider_api_key=args.provider_api_key,
        model=args.embed_model,
    )
    embedding = embedder.embed_text(args.query)

    with connect(database_url(args.database_url)) as conn:
        acquire_corpus_read_lock(conn)
        validate_embedding_profile(
            conn,
            expected_model=resolve_embedding_index_id(embedder),
            expected_dimension=len(embedding),
        )
        rows = hybrid_search(
            conn,
            query=args.query,
            embedding=embedding,
            limit=args.limit,
            vector_candidates=args.vector_candidates,
            text_candidates=args.text_candidates,
        )

    print(f"Query: {args.query}")
    for index, row in enumerate(rows, start=1):
        snippet = " ".join(str(row["raw_text"]).split())[:420]
        print(
            f"{index}. hybrid={float(row['hybrid_score']):.5f} "
            f"vector={float(row['vector_score']):.4f} text={float(row['text_score']):.4f}"
        )
        print(f"   {row['citation_label']}")
        print(f"   {snippet}")

    if not rows:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
