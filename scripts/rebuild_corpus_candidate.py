from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import sys
from typing import Any
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from document_search.chunker import DEFAULT_MAX_CHARS, chunk_document
from document_search.extractor import extract_docx
from corpus_candidate_integrity import (
    CANDIDATE_SCHEMA_VERSION,
    CANDIDATE_TOOL_VERSION,
    build_integrity_manifest,
    verify_candidate_integrity,
)


def discover_documents(docs_dir: Path) -> list[Path]:
    return sorted(
        path.resolve()
        for path in docs_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".docx"
        and not path.name.startswith("~$")
    )


def _require_nonempty(value: object, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} is empty")
    return text


def validate_candidate(
    records: list[tuple[Path, dict[str, Any], dict[str, Any]]],
    *,
    expected_documents: int | None = None,
) -> dict[str, int]:
    if not records:
        raise ValueError("No DOCX documents were discovered")
    if expected_documents is not None and len(records) != expected_documents:
        raise ValueError(
            f"Document count mismatch: expected {expected_documents}, found {len(records)}"
        )

    document_ids: set[str] = set()
    chunk_ids: set[str] = set()
    total_blocks = 0
    total_chunks = 0
    total_characters = 0

    for source_path, extracted, chunked in records:
        metadata = extracted.get("metadata") or {}
        doc_id = _require_nonempty(metadata.get("doc_id"), f"{source_path}: doc_id")
        if doc_id in document_ids:
            raise ValueError(f"Duplicate document ID: {doc_id}")
        document_ids.add(doc_id)

        blocks = extracted.get("blocks") or []
        if extracted.get("block_count") != len(blocks):
            raise ValueError(f"{doc_id}: extraction block_count does not match blocks")
        substantive_blocks = [block for block in blocks if str(block.get("text") or "").strip()]
        if not substantive_blocks:
            raise ValueError(f"{doc_id}: extraction contains no substantive blocks")

        block_ids = [_require_nonempty(block.get("block_id"), f"{doc_id}: block_id") for block in blocks]
        if len(block_ids) != len(set(block_ids)):
            raise ValueError(f"{doc_id}: extraction contains duplicate block IDs")
        known_block_ids = set(block_ids)
        expected_block_ids = {
            _require_nonempty(block.get("block_id"), f"{doc_id}: block_id")
            for block in substantive_blocks
        }

        chunk_metadata = chunked.get("metadata") or {}
        if chunk_metadata.get("doc_id") != doc_id:
            raise ValueError(f"{doc_id}: chunk metadata belongs to another document")
        chunks = chunked.get("chunks") or []
        if chunked.get("chunk_count") != len(chunks):
            raise ValueError(f"{doc_id}: chunk_count does not match chunks")
        if not chunks:
            raise ValueError(f"{doc_id}: no chunks were produced")

        covered_block_ids: set[str] = set()
        for chunk in chunks:
            chunk_id = _require_nonempty(chunk.get("chunk_id"), f"{doc_id}: chunk_id")
            if chunk_id in chunk_ids:
                raise ValueError(f"Duplicate chunk ID: {chunk_id}")
            chunk_ids.add(chunk_id)
            if chunk.get("doc_id") != doc_id:
                raise ValueError(f"{chunk_id}: chunk belongs to another document")

            raw_text = _require_nonempty(chunk.get("raw_text"), f"{chunk_id}: raw_text")
            _require_nonempty(chunk.get("searchable_text"), f"{chunk_id}: searchable_text")
            if chunk.get("char_count") != len(raw_text):
                raise ValueError(f"{chunk_id}: char_count does not match raw_text")

            for block_id in chunk.get("block_ids") or []:
                block_id = _require_nonempty(block_id, f"{chunk_id}: block_id")
                if block_id != "metadata" and block_id not in known_block_ids:
                    raise ValueError(f"{chunk_id}: unknown block ID {block_id}")
                if block_id != "metadata":
                    covered_block_ids.add(block_id)

        missing = sorted(expected_block_ids - covered_block_ids)
        if missing:
            preview = ", ".join(missing[:10])
            raise ValueError(f"{doc_id}: {len(missing)} substantive blocks are not chunked: {preview}")

        total_blocks += len(substantive_blocks)
        total_chunks += len(chunks)
        total_characters += sum(len(str(block.get("text") or "")) for block in substantive_blocks)

    return {
        "documents": len(records),
        "substantive_blocks": total_blocks,
        "chunks": total_chunks,
        "characters": total_characters,
    }


def build_candidate(
    docs_dir: Path,
    output_dir: Path,
    *,
    expected_documents: int | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> dict[str, Any]:
    docs_dir = docs_dir.resolve()
    output_dir = output_dir.resolve()
    if not docs_dir.is_dir():
        raise ValueError(f"Documents directory does not exist: {docs_dir}")
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"Output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise ValueError(f"Output directory is not empty: {output_dir}")

    sources = discover_documents(docs_dir)
    records: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    for source_path in sources:
        extracted = extract_docx(source_path).to_dict()
        chunked = chunk_document(extracted, max_chars=max_chars)
        records.append((source_path, extracted, chunked))

    summary = validate_candidate(records, expected_documents=expected_documents)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.parent / f".{output_dir.name}.staging-{uuid4().hex}"

    try:
        extracted_dir = staging_dir / "extracted"
        chunks_dir = staging_dir / "chunks"
        extracted_dir.mkdir(parents=True)
        chunks_dir.mkdir(parents=True)
        extraction_manifest: list[dict[str, Any]] = []
        chunk_manifest: list[dict[str, Any]] = []
        extracted_files: list[Path] = []
        chunk_files: list[Path] = []

        for source_path, extracted, chunked in records:
            metadata = extracted["metadata"]
            doc_id = str(metadata["doc_id"])
            extracted_name = f"{doc_id}.json"
            chunks_name = f"{doc_id}.chunks.json"
            staged_extracted_path = extracted_dir / extracted_name
            staged_chunks_path = chunks_dir / chunks_name
            final_extracted_path = Path("extracted") / extracted_name
            final_chunks_path = Path("chunks") / chunks_name

            staged_extracted_path.write_text(
                json.dumps(extracted, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            staged_chunks_path.write_text(
                json.dumps(chunked, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            extracted_files.append(staged_extracted_path)
            chunk_files.append(staged_chunks_path)
            extraction_manifest.append(
                {
                    "source_name": metadata.get("source_name"),
                    "source_path": str(source_path),
                    "doc_id": doc_id,
                    "output_path": str(final_extracted_path),
                }
            )
            chunk_manifest.append(
                {
                    "doc_id": doc_id,
                    "source_name": metadata.get("source_name"),
                    "extracted_path": str(final_extracted_path),
                    "chunked_path": str(final_chunks_path),
                    "chunks": chunked["chunk_count"],
                    "max_chars": max_chars,
                }
            )

        (extracted_dir / "manifest.json").write_text(
            json.dumps(extraction_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (chunks_dir / "manifest.json").write_text(
            json.dumps(chunk_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary = {
            **summary,
            "schema_version": CANDIDATE_SCHEMA_VERSION,
            "tool_version": CANDIDATE_TOOL_VERSION,
            "integrity": build_integrity_manifest(
                docs_dir=docs_dir,
                candidate_dir=staging_dir,
                source_files=sources,
                extracted_files=extracted_files,
                chunk_files=chunk_files,
            ),
        }
        verify_candidate_integrity(staging_dir, summary)
        report = {
            "status": "ready",
            "created_at": datetime.now(UTC).isoformat(),
            "docs_dir": str(docs_dir),
            "max_chars": max_chars,
            **summary,
        }
        (staging_dir / "candidate_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (staging_dir / "READY").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        if output_dir.exists():
            output_dir.rmdir()
        staging_dir.rename(output_dir)
    except BaseException:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and validate a complete corpus candidate without touching the live index."
    )
    parser.add_argument("--docs-dir", required=True, help="Directory containing source DOCX files.")
    parser.add_argument("--output-dir", required=True, help="New empty directory for the candidate.")
    parser.add_argument("--expected-documents", type=int, default=None)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    args = parser.parse_args()

    try:
        summary = build_candidate(
            Path(args.docs_dir),
            Path(args.output_dir),
            expected_documents=args.expected_documents,
            max_chars=args.max_chars,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Candidate ready: {Path(args.output_dir).resolve()}")
    print(
        f"Documents: {summary['documents']}; blocks: {summary['substantive_blocks']}; "
        f"chunks: {summary['chunks']}; characters: {summary['characters']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
