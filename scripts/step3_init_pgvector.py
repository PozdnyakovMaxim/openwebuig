from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from document_search.pgvector_store import connect, database_url, init_schema, redact_url
from document_search.provider_api import make_embedder


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 3: initialize PostgreSQL + pgvector schema.")
    parser.add_argument("--database-url", default=None, help="PostgreSQL URL. Defaults to DATABASE_URL env.")
    parser.add_argument("--embedding-dim", type=int, default=1024, help="Vector dimension.")
    parser.add_argument("--embed-provider", default=None, help="Embedding provider: local or provider.")
    parser.add_argument("--provider-api-base-url", default=None, help="Compatible provider base URL.")
    parser.add_argument("--provider-api-key", default=None, help="Provider API key. Prefer env PROVIDER_API_KEY.")
    parser.add_argument("--embed-model", default=None, help="Embedding model name.")
    parser.add_argument("--no-embedding-check", action="store_true", help="Skip embedding dimension check.")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate document search tables.")
    args = parser.parse_args()

    if not args.no_embedding_check:
        embedder = make_embedder(
            provider=args.embed_provider,
            provider_api_base_url=args.provider_api_base_url,
            provider_api_key=args.provider_api_key,
            model=args.embed_model,
        )
        actual_dim = embedder.embedding_dimension()
        if actual_dim != args.embedding_dim:
            raise SystemExit(
                f"Embedding dimension mismatch: model returned {actual_dim}, "
                f"but --embedding-dim is {args.embedding_dim}."
            )
        print(f"Embedding model {embedder.model}: dimension {actual_dim}")

    url = database_url(args.database_url)
    with connect(url) as conn:
        warnings = init_schema(conn, embedding_dim=args.embedding_dim, recreate=args.recreate)

    for warning in warnings:
        print(f"WARNING: {warning}")
    if any("IVFFLAT index was not created either" in warning for warning in warnings):
        print(f"pgvector schema has no ANN index: {redact_url(url)}")
        return 1
    print(f"pgvector schema is ready: {redact_url(url)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
