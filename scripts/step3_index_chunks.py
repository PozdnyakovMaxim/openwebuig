from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from document_search.pgvector_store import (
    connect,
    database_url,
    delete_chunks_for_doc,
    init_schema,
    redact_url,
    upsert_chunk,
    upsert_document,
)
from document_search.provider_api import make_embedder


def resolve_chunk_files(input_dir: Path) -> list[Path]:
    manifest_path = input_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        paths: list[Path] = []
        for item in manifest:
            path = Path(item["chunked_path"])
            if path.exists():
                paths.append(path.resolve())
                continue
            fallback = input_dir / path.name
            if fallback.exists():
                paths.append(fallback.resolve())
        return paths
    return sorted(path for path in input_dir.glob("*.chunks.json"))


def batched(items: list[dict], size: int) -> list[list[dict]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 3: embed chunks and load them into pgvector.")
    parser.add_argument("--input-dir", required=True, help="Directory produced by step2_chunk_corpus.py.")
    parser.add_argument("--database-url", default=None, help="PostgreSQL URL. Defaults to DATABASE_URL env.")
    parser.add_argument("--embed-provider", default=None, help="Embedding provider: local or provider.")
    parser.add_argument("--provider-api-base-url", default=None, help="Compatible provider base URL.")
    parser.add_argument("--provider-api-key", default=None, help="Provider API key. Prefer env PROVIDER_API_KEY.")
    parser.add_argument("--embed-model", default=None, help="Embedding model name.")
    parser.add_argument("--embedding-dim", type=int, default=1024, help="Vector dimension.")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of chunks per embedding batch.")
    parser.add_argument("--limit", type=int, default=None, help="Index only first N chunks per document for smoke tests.")
    parser.add_argument("--no-replace", action="store_true", help="Do not delete existing chunks for each document.")
    parser.add_argument("--init-schema", action="store_true", help="Create schema before indexing.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    files = resolve_chunk_files(input_dir)
    if not files:
        raise SystemExit(f"No chunk files found in {input_dir}")

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

    total_chunks = 0
    url = database_url(args.database_url)
    with connect(url) as conn:
        if args.init_schema:
            init_schema(conn, embedding_dim=args.embedding_dim)

        for path in files:
            data = json.loads(path.read_text(encoding="utf-8"))
            metadata = data["metadata"]
            chunks = data.get("chunks") or []
            if args.limit is not None:
                chunks = chunks[: args.limit]

            doc_id = metadata["doc_id"]
            upsert_document(conn, metadata)
            if not args.no_replace:
                delete_chunks_for_doc(conn, doc_id)

            for batch in batched(chunks, args.batch_size):
                texts = [chunk["searchable_text"] for chunk in batch]
                embeddings = embedder.embed_texts(texts)
                for chunk, embedding in zip(batch, embeddings, strict=True):
                    upsert_chunk(conn, chunk, embedding=embedding, embedding_model=embedder.model)
                total_chunks += len(batch)
                print(f"Indexed {total_chunks} chunks...", flush=True)

            conn.commit()
            print(f"Document indexed: {doc_id} ({len(chunks)} chunks)")

    print(f"Indexed {total_chunks} chunks into {redact_url(url)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
