from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from document_search.pgvector_store import (
    CORPUS_PROMOTION_ADVISORY_LOCK_KEY,
    acquire_corpus_promotion_lock,
    connect,
    count_rows,
    database_url,
    delete_chunks_for_doc,
    init_schema,
    redact_url,
    resolve_embedding_index_id,
    upsert_chunk,
    upsert_document,
)
from document_search.provider_api import make_embedder
from corpus_candidate_integrity import (
    validate_integrity_metadata,
    verify_candidate_audit,
    verify_candidate_integrity,
)


def _candidate_chunk_hashes(
    input_dir: Path,
    marker: dict[str, object],
) -> dict[Path, str]:
    validate_integrity_metadata(marker)
    candidate_dir = input_dir.parent.resolve()
    resolved_input_dir = input_dir.resolve()
    integrity = marker["integrity"]
    assert isinstance(integrity, dict)
    entries = integrity["files"]
    assert isinstance(entries, list)
    hashes: dict[Path, str] = {}
    for entry in entries:
        assert isinstance(entry, dict)
        if entry.get("kind") != "chunk_json":
            continue
        relative_path = Path(str(entry["path"]))
        path = (candidate_dir / relative_path).resolve()
        if not path.is_relative_to(resolved_input_dir):
            raise ValueError(
                f"Candidate chunk path is outside the selected input directory: {relative_path}"
            )
        if path in hashes:
            raise ValueError(f"Candidate contains a duplicate chunk path: {relative_path}")
        hashes[path] = str(entry["sha256"]).casefold()
    if not hashes:
        raise ValueError("Candidate integrity metadata contains no chunk JSON files")
    return hashes


def resolve_chunk_files(
    input_dir: Path,
    *,
    candidate_marker: dict[str, object] | None = None,
) -> list[Path]:
    input_dir = input_dir.resolve()
    manifest_path = input_dir / "manifest.json"
    if candidate_marker is not None:
        expected = _candidate_chunk_hashes(input_dir, candidate_marker)
        if not manifest_path.is_file():
            raise ValueError(f"Candidate chunk manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, list):
            raise ValueError("Candidate chunk manifest must be a JSON list")
        paths: list[Path] = []
        candidate_dir = input_dir.parent
        for item in manifest:
            if not isinstance(item, dict):
                raise ValueError("Candidate chunk manifest contains a non-object entry")
            raw_value = str(item.get("chunked_path") or "")
            raw_path = Path(raw_value)
            if not raw_value or raw_path.is_absolute() or ".." in raw_path.parts:
                raise ValueError(
                    "Candidate chunked_path must be relative to and contained in the candidate: "
                    f"{raw_value!r}"
                )
            path = (candidate_dir / raw_path).resolve()
            if not path.is_relative_to(input_dir):
                raise ValueError(f"Candidate manifest path is outside chunks/: {raw_path}")
            paths.append(path)
        if len(paths) != len(set(paths)):
            raise ValueError("Candidate chunk manifest contains duplicate paths")
        if set(paths) != set(expected):
            missing = sorted(str(path) for path in set(expected) - set(paths))
            unexpected = sorted(str(path) for path in set(paths) - set(expected))
            raise ValueError(
                "Candidate chunk manifest does not match READY integrity entries: "
                f"missing={missing}, unexpected={unexpected}"
            )
        return sorted(paths)
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


def load_candidate_marker(input_dir: Path) -> dict[str, object]:
    ready_path = input_dir.parent / "READY"
    if not ready_path.is_file():
        raise ValueError(
            f"Validated candidate marker is missing: {ready_path}. "
            "Build the corpus with rebuild_corpus_candidate.py first."
        )
    try:
        data = json.loads(ready_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid candidate marker: {ready_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Candidate marker must be a JSON object: {ready_path}")

    for field in ("documents", "substantive_blocks", "chunks", "characters"):
        value = data.get(field)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"Candidate marker has invalid {field}: {value!r}")
    validate_integrity_metadata(data)
    return data


def batched(items: list[dict], size: int) -> list[list[dict]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def load_and_validate_corpus(
    files: list[Path],
    *,
    limit: int | None = None,
    expected_documents: int | None = None,
    expected_file_hashes: dict[Path, str] | None = None,
) -> list[dict]:
    if expected_documents is not None and len(files) != expected_documents:
        raise ValueError(
            f"Document count mismatch: expected {expected_documents}, found {len(files)}"
        )

    records: list[dict] = []
    document_ids: set[str] = set()
    chunk_ids: set[str] = set()
    for path in files:
        resolved_path = path.resolve()
        payload = path.read_bytes()
        if expected_file_hashes is not None:
            expected_hash = expected_file_hashes.get(resolved_path)
            if expected_hash is None:
                raise ValueError(f"Loaded chunk file is not tracked by READY: {resolved_path}")
            actual_hash = hashlib.sha256(payload).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Loaded chunk file hash mismatch for {resolved_path}: "
                    f"expected {expected_hash}, received {actual_hash}"
                )
        data = json.loads(payload)
        metadata = data.get("metadata") or {}
        doc_id = str(metadata.get("doc_id") or "").strip()
        source_name = str(metadata.get("source_name") or "").strip()
        if not doc_id:
            raise ValueError(f"Missing doc_id in {path}")
        if not source_name:
            raise ValueError(f"Missing source_name in {path}")
        if doc_id in document_ids:
            raise ValueError(f"Duplicate document ID: {doc_id}")
        document_ids.add(doc_id)

        all_chunks = data.get("chunks") or []
        if data.get("chunk_count") != len(all_chunks):
            raise ValueError(f"{doc_id}: chunk_count does not match chunks")
        chunks = all_chunks if limit is None else all_chunks[:limit]
        if not chunks:
            raise ValueError(f"{doc_id}: no chunks selected for indexing")

        for chunk in chunks:
            chunk_id = str(chunk.get("chunk_id") or "").strip()
            if not chunk_id:
                raise ValueError(f"{doc_id}: chunk without chunk_id")
            if chunk_id in chunk_ids:
                raise ValueError(f"Duplicate chunk ID: {chunk_id}")
            chunk_ids.add(chunk_id)
            if chunk.get("doc_id") != doc_id:
                raise ValueError(f"{chunk_id}: chunk belongs to another document")
            searchable_text = str(chunk.get("searchable_text") or "").strip()
            raw_text = str(chunk.get("raw_text") or "").strip()
            if not searchable_text or not raw_text:
                raise ValueError(f"{chunk_id}: chunk text is empty")
            if chunk.get("char_count") != len(str(chunk.get("raw_text") or "")):
                raise ValueError(f"{chunk_id}: char_count does not match raw_text")

        records.append({"path": resolved_path, "metadata": metadata, "chunks": chunks})
    if expected_file_hashes is not None and {
        record["path"] for record in records
    } != set(expected_file_hashes):
        raise ValueError("Loaded chunk files do not exactly match READY integrity entries")
    return records


def embed_corpus(records: list[dict], embedder: object, *, batch_size: int) -> int:
    total = 0
    for record in records:
        embedded_chunks: list[tuple[dict, list[float]]] = []
        for batch in batched(record["chunks"], batch_size):
            texts = [chunk["searchable_text"] for chunk in batch]
            embeddings = embedder.embed_texts(texts)
            if len(embeddings) != len(batch):
                raise ValueError(
                    f"Embedding count mismatch for {record['metadata']['doc_id']}: "
                    f"expected {len(batch)}, received {len(embeddings)}"
                )
            embedded_chunks.extend(zip(batch, embeddings, strict=True))
            total += len(batch)
            print(f"Prepared {total} embeddings...", flush=True)
        record["embedded_chunks"] = embedded_chunks
    return total


def validate_embeddings(records: list[dict], *, embedding_dim: int) -> None:
    for record in records:
        for chunk, embedding in record.get("embedded_chunks") or []:
            if len(embedding) != embedding_dim:
                raise ValueError(
                    f"{chunk['chunk_id']}: expected embedding dimension {embedding_dim}, "
                    f"received {len(embedding)}"
                )


def index_records(
    conn: object,
    records: list[dict],
    *,
    embedding_model: str,
    replace_documents: bool,
) -> int:
    total = 0
    for record in records:
        metadata = record["metadata"]
        doc_id = metadata["doc_id"]
        upsert_document(conn, metadata)
        if replace_documents:
            delete_chunks_for_doc(conn, doc_id)
        for chunk, embedding in record["embedded_chunks"]:
            upsert_chunk(
                conn,
                chunk,
                embedding=embedding,
                embedding_model=embedding_model,
            )
            total += 1
        print(f"Document prepared for commit: {doc_id} ({len(record['chunks'])} chunks)")
    return total


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
    parser.add_argument(
        "--atomic",
        action="store_true",
        help="Prepare every embedding first and commit all selected documents in one transaction.",
    )
    parser.add_argument(
        "--replace-corpus",
        action="store_true",
        help="Atomically replace the complete corpus. Requires a validated candidate and --atomic.",
    )
    parser.add_argument("--expected-documents", type=int, default=None)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.replace_corpus and not args.atomic:
        raise SystemExit("--replace-corpus requires --atomic")
    if args.replace_corpus and args.no_replace:
        raise SystemExit("--replace-corpus cannot be combined with --no-replace")
    if args.replace_corpus and args.limit is not None:
        raise SystemExit("--replace-corpus cannot be combined with --limit")
    if args.replace_corpus and args.expected_documents is None:
        raise SystemExit("--replace-corpus requires --expected-documents")

    input_dir = Path(args.input_dir).resolve()
    candidate_summary: dict[str, object] | None = None
    expected_file_hashes: dict[Path, str] | None = None
    if args.replace_corpus:
        try:
            candidate_summary = load_candidate_marker(input_dir)
            verify_candidate_integrity(input_dir.parent, candidate_summary)
            verify_candidate_audit(input_dir.parent, candidate_summary)
            expected_file_hashes = _candidate_chunk_hashes(input_dir, candidate_summary)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(str(exc)) from exc
        if candidate_summary["documents"] != args.expected_documents:
            raise SystemExit(
                "Candidate document count does not match --expected-documents: "
                f"{candidate_summary['documents']} != {args.expected_documents}"
            )
    try:
        files = resolve_chunk_files(input_dir, candidate_marker=candidate_summary)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Chunk file resolution failed: {exc}") from exc
    if not files:
        raise SystemExit(f"No chunk files found in {input_dir}")

    try:
        records = load_and_validate_corpus(
            files,
            limit=args.limit,
            expected_documents=args.expected_documents,
            expected_file_hashes=expected_file_hashes,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Corpus validation failed: {exc}") from exc

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
    try:
        embedding_model_id = resolve_embedding_index_id(embedder)
    except ValueError as exc:
        raise SystemExit(f"Embedding profile is invalid: {exc}") from exc

    if args.atomic:
        try:
            prepared_chunks = embed_corpus(records, embedder, batch_size=args.batch_size)
            validate_embeddings(records, embedding_dim=args.embedding_dim)
        except (RuntimeError, ValueError) as exc:
            raise SystemExit(f"Embedding preparation failed before database changes: {exc}") from exc

    total_chunks = 0
    url = database_url(args.database_url)
    with connect(url) as conn:
        if args.init_schema:
            schema_warnings = init_schema(conn, embedding_dim=args.embedding_dim)
            for warning in schema_warnings:
                print(f"WARNING: {warning}")
            if any("IVFFLAT index was not created either" in warning for warning in schema_warnings):
                raise SystemExit("Schema initialization failed to create a vector ANN index")

        if args.atomic:
            expected_chunks = sum(len(record["chunks"]) for record in records)
            if candidate_summary is not None and candidate_summary["chunks"] != expected_chunks:
                raise SystemExit(
                    "Candidate chunk count does not match validated artifacts: "
                    f"{candidate_summary['chunks']} != {expected_chunks}"
                )
            if prepared_chunks != expected_chunks:
                raise SystemExit(
                    f"Prepared chunk count mismatch: expected {expected_chunks}, "
                    f"received {prepared_chunks}"
                )
            try:
                with conn.transaction():
                    if args.replace_corpus:
                        acquire_corpus_promotion_lock(conn)
                        try:
                            verify_candidate_integrity(input_dir.parent, candidate_summary)
                            verify_candidate_audit(input_dir.parent, candidate_summary)
                        except (OSError, ValueError) as exc:
                            raise ValueError(
                                "Candidate integrity verification failed under promotion lock: "
                                f"{exc}"
                            ) from exc
                        conn.execute("DELETE FROM doc_documents")
                    total_chunks = index_records(
                        conn,
                        records,
                        embedding_model=embedding_model_id,
                        replace_documents=not args.no_replace and not args.replace_corpus,
                    )
                    if total_chunks != expected_chunks:
                        raise ValueError(
                            f"Indexed chunk count mismatch: expected {expected_chunks}, "
                            f"received {total_chunks}"
                        )
                    if args.replace_corpus:
                        verify_candidate_integrity(input_dir.parent, candidate_summary)
                        verify_candidate_audit(input_dir.parent, candidate_summary)
                        counts = count_rows(conn)
                        if counts != {
                            "documents": len(records),
                            "chunks": expected_chunks,
                        }:
                            raise ValueError(
                                "Database count mismatch before commit: "
                                f"expected {len(records)} documents/{expected_chunks} chunks, "
                                f"found {counts['documents']} documents/{counts['chunks']} chunks"
                            )
            except Exception as exc:
                raise SystemExit(f"Atomic indexing rolled back: {exc}") from exc

            print(f"Atomic commit completed for {len(records)} documents.")
            print(f"Indexed {total_chunks} chunks into {redact_url(url)}")
            return 0

        for record in records:
            metadata = record["metadata"]
            chunks = record["chunks"]
            doc_id = metadata["doc_id"]
            upsert_document(conn, metadata)
            if not args.no_replace:
                delete_chunks_for_doc(conn, doc_id)

            for batch in batched(chunks, args.batch_size):
                texts = [chunk["searchable_text"] for chunk in batch]
                embeddings = embedder.embed_texts(texts)
                for chunk, embedding in zip(batch, embeddings, strict=True):
                    upsert_chunk(
                        conn,
                        chunk,
                        embedding=embedding,
                        embedding_model=embedding_model_id,
                    )
                total_chunks += len(batch)
                print(f"Indexed {total_chunks} chunks...", flush=True)

            conn.commit()
            print(f"Document indexed: {doc_id} ({len(chunks)} chunks)")

    print(f"Indexed {total_chunks} chunks into {redact_url(url)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
