from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any

from .chunker import DEFAULT_MAX_CHARS, chunk_document
from .extractor import extract_docx
from .pgvector_store import connect, database_url, redact_url


WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+", re.UNICODE)


def audit_corpus(
    docs_dir: str | Path,
    extracted_dir: str | Path,
    chunks_dir: str | Path,
    *,
    database: str | None = None,
    skip_database: bool = False,
) -> dict[str, Any]:
    docs_path = Path(docs_dir).expanduser().resolve()
    extracted_path = Path(extracted_dir).expanduser().resolve()
    chunks_path = Path(chunks_dir).expanduser().resolve()
    issues: list[dict[str, Any]] = []

    for label, path in (
        ("docs", docs_path),
        ("extracted", extracted_path),
        ("chunks", chunks_path),
    ):
        if not path.is_dir():
            _issue(
                issues,
                "error",
                f"{label}_directory_missing",
                f"Directory does not exist: {path}",
            )

    source_files = _source_files(docs_path)
    if docs_path.is_dir() and not source_files:
        _issue(
            issues,
            "error",
            "source_documents_missing",
            f"No DOCX source documents found in {docs_path}",
        )
    source_by_name = _group_by_name(source_files)
    for name, paths in source_by_name.items():
        if len(paths) > 1:
            _issue(
                issues,
                "error",
                "duplicate_source_name",
                f"Multiple source files have the same name: {name}",
                source_name=name,
                details={"paths": [str(path) for path in paths]},
            )

    extracted_manifest = _load_manifest(extracted_path, issues, "extraction")
    chunks_manifest = _load_manifest(chunks_path, issues, "chunks")
    _audit_artifact_inventory(
        extracted_path,
        extracted_manifest,
        issues,
        stage="extraction",
        path_field="output_path",
        fallback_suffix=".json",
    )
    _audit_artifact_inventory(
        chunks_path,
        chunks_manifest,
        issues,
        stage="chunks",
        path_field="chunked_path",
        fallback_suffix=".chunks.json",
    )
    extracted_records = _load_extracted_records(extracted_manifest, extracted_path, issues)
    chunk_records = _load_chunk_records(chunks_manifest, chunks_path, issues)

    extracted_by_name = _group_records(extracted_records, "source_name")
    extracted_by_doc_id = _group_records(extracted_records, "doc_id")
    chunks_by_doc_id = _group_records(chunk_records, "doc_id")
    documents: list[dict[str, Any]] = []
    expected_db_documents: dict[str, dict[str, Any]] = {}
    expected_db_chunks: dict[str, dict[str, Any]] = {}

    for source_file in source_files:
        source_name = source_file.name
        matches = extracted_by_name.get(_key(source_name), [])
        if len(matches) != 1:
            if not matches:
                _issue(
                    issues,
                    "error",
                    "source_not_extracted",
                    f"Source document is absent from extraction manifest: {source_name}",
                    source_name=source_name,
                )
            else:
                _issue(
                    issues,
                    "error",
                    "duplicate_extraction_source",
                    f"Extraction manifest contains multiple records for: {source_name}",
                    source_name=source_name,
                    details={"doc_ids": [record.get("doc_id") for record in matches]},
                )
            continue

        extracted_record = matches[0]
        data = extracted_record["data"]
        metadata = data.get("metadata") or {}
        doc_id = str(metadata.get("doc_id") or extracted_record.get("doc_id") or "")
        chunk_matches = chunks_by_doc_id.get(_key(doc_id), [])
        document_report = _audit_document(
            source_file,
            extracted_record,
            chunk_matches,
            issues,
        )
        documents.append(document_report)
        if doc_id:
            expected_db_documents[doc_id] = {
                "source_name": source_name,
                "metadata": metadata,
            }
        for chunk_record in chunk_matches:
            for chunk in chunk_record["data"].get("chunks") or []:
                chunk_id = str(chunk.get("chunk_id") or "")
                if chunk_id:
                    if chunk_id in expected_db_chunks:
                        _issue(
                            issues,
                            "error",
                            "duplicate_chunk_id",
                            f"Duplicate chunk_id in artifacts: {chunk_id}",
                            doc_id=doc_id,
                            source_name=source_name,
                        )
                    expected_db_chunks[chunk_id] = chunk

    source_names = set(source_by_name)
    for record in extracted_records:
        source_name = str(record.get("source_name") or "")
        if _key(source_name) not in source_names:
            _issue(
                issues,
                "error",
                "extracted_source_missing",
                f"Extraction artifact has no source DOCX: {source_name}",
                doc_id=record.get("doc_id"),
                source_name=source_name,
            )

    for doc_id, records in extracted_by_doc_id.items():
        if doc_id and len(records) > 1:
            _issue(
                issues,
                "error",
                "duplicate_doc_id",
                f"Multiple extraction records use doc_id: {records[0].get('doc_id')}",
                doc_id=records[0].get("doc_id"),
                details={"sources": [record.get("source_name") for record in records]},
            )

    extracted_doc_ids = set(extracted_by_doc_id)
    for record in chunk_records:
        doc_id = str(record.get("doc_id") or "")
        if _key(doc_id) not in extracted_doc_ids:
            _issue(
                issues,
                "error",
                "chunked_document_missing_extraction",
                f"Chunk artifact has no extraction record: {doc_id}",
                doc_id=doc_id,
                source_name=record.get("source_name"),
            )

    database_report: dict[str, Any] = {"checked": False}
    if not skip_database:
        database_report = _audit_database(
            database,
            expected_db_documents,
            expected_db_chunks,
            issues,
        )

    error_count = sum(issue["level"] == "error" for issue in issues)
    warning_count = sum(issue["level"] == "warning" for issue in issues)
    status = "error" if error_count else "warning" if warning_count else "ok"
    corpus_totals = {
        "source_characters": sum(int(item.get("source_chars") or 0) for item in documents),
        "extracted_characters": sum(int(item.get("extracted_chars") or 0) for item in documents),
        "chunk_characters": sum(int(item.get("chunk_chars") or 0) for item in documents),
        "extracted_blocks": sum(int(item.get("blocks") or 0) for item in documents),
        "indexable_blocks": sum(int(item.get("indexable_blocks") or 0) for item in documents),
        "covered_blocks": sum(int(item.get("covered_blocks") or 0) for item in documents),
        "intentionally_non_indexed_blocks": sum(
            int(item.get("intentionally_non_indexed_blocks") or 0) for item in documents
        ),
        "unsearchable_non_indexed_blocks": sum(
            int(item.get("unsearchable_non_indexed_blocks") or 0) for item in documents
        ),
    }
    return {
        "status": status,
        "summary": {
            "source_documents": len(source_files),
            "extracted_documents": len(extracted_records),
            "chunked_documents": len(chunk_records),
            "artifact_chunks": len(expected_db_chunks),
            "errors": error_count,
            "warnings": warning_count,
            **corpus_totals,
        },
        "paths": {
            "docs": str(docs_path),
            "extracted": str(extracted_path),
            "chunks": str(chunks_path),
        },
        "database": database_report,
        "documents": documents,
        "issues": issues,
    }


def _source_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path.resolve()
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.casefold() == ".docx"
        and not path.name.startswith("~$")
    )


def _load_manifest(
    directory: Path,
    issues: list[dict[str, Any]],
    stage: str,
) -> list[dict[str, Any]]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        _issue(
            issues,
            "error",
            f"{stage}_manifest_missing",
            f"Missing {stage} manifest: {manifest_path}",
        )
        return []
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _issue(
            issues,
            "error",
            f"{stage}_manifest_invalid",
            f"Cannot read {stage} manifest: {exc}",
        )
        return []
    if not isinstance(data, list):
        _issue(
            issues,
            "error",
            f"{stage}_manifest_invalid",
            f"{stage.capitalize()} manifest must contain a JSON array",
        )
        return []
    valid_items = [item for item in data if isinstance(item, dict)]
    invalid_count = len(data) - len(valid_items)
    if invalid_count:
        _issue(
            issues,
            "error",
            f"{stage}_manifest_entries_invalid",
            f"{invalid_count} {stage} manifest entries are not JSON objects",
        )
    return valid_items


def _audit_artifact_inventory(
    directory: Path,
    manifest: list[dict[str, Any]],
    issues: list[dict[str, Any]],
    *,
    stage: str,
    path_field: str,
    fallback_suffix: str,
) -> None:
    if not directory.is_dir():
        return
    referenced: set[Path] = set()
    for item in manifest:
        doc_id = str(item.get("doc_id") or "")
        referenced.add(
            _resolve_artifact(
                directory,
                item.get(path_field),
                f"{doc_id}{fallback_suffix}",
            )
        )
    actual = {
        path.resolve()
        for path in directory.glob("*.json")
        if path.name != "manifest.json"
    }
    orphaned = sorted(actual - referenced)
    if orphaned:
        _issue(
            issues,
            "error",
            f"{stage}_artifacts_orphaned",
            f"{len(orphaned)} {stage} JSON files are not referenced by manifest",
            details={"paths": [str(path) for path in orphaned[:50]]},
        )


def _resolve_artifact(directory: Path, raw_path: Any, fallback_name: str) -> Path:
    candidate = Path(str(raw_path or "")).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    if candidate.name:
        fallback = directory / candidate.name
        if fallback.is_file():
            return fallback.resolve()
    return (directory / fallback_name).resolve()


def _load_extracted_records(
    manifest: list[dict[str, Any]],
    directory: Path,
    issues: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in manifest:
        doc_id = str(item.get("doc_id") or "")
        path = _resolve_artifact(directory, item.get("output_path"), f"{doc_id}.json")
        data = _read_json(path, issues, "extracted_json_invalid", doc_id, item.get("source_name"))
        if data is None:
            continue
        metadata = data.get("metadata") or {}
        blocks = data.get("blocks") or []
        artifact_doc_id = str(metadata.get("doc_id") or "")
        artifact_source_name = str(metadata.get("source_name") or "")
        if doc_id and artifact_doc_id and doc_id != artifact_doc_id:
            _issue(
                issues,
                "error",
                "extraction_manifest_doc_id_mismatch",
                f"Manifest doc_id differs from extraction JSON: {path.name}",
                doc_id=artifact_doc_id,
                source_name=artifact_source_name or item.get("source_name"),
                details={"manifest_doc_id": doc_id, "artifact_doc_id": artifact_doc_id},
            )
        manifest_source_name = str(item.get("source_name") or "")
        if manifest_source_name and artifact_source_name and manifest_source_name != artifact_source_name:
            _issue(
                issues,
                "error",
                "extraction_manifest_source_mismatch",
                f"Manifest source_name differs from extraction JSON: {path.name}",
                doc_id=artifact_doc_id or doc_id,
                source_name=artifact_source_name,
                details={
                    "manifest_source_name": manifest_source_name,
                    "artifact_source_name": artifact_source_name,
                },
            )
        if data.get("block_count") != len(blocks):
            _issue(
                issues,
                "error",
                "extracted_block_count_mismatch",
                f"block_count differs from blocks length: {path.name}",
                doc_id=metadata.get("doc_id") or doc_id,
                source_name=metadata.get("source_name") or item.get("source_name"),
            )
        records.append(
            {
                "doc_id": metadata.get("doc_id") or doc_id,
                "source_name": metadata.get("source_name") or item.get("source_name"),
                "path": path,
                "data": data,
                "max_chars": item.get("max_chars"),
            }
        )
    return records


def _load_chunk_records(
    manifest: list[dict[str, Any]],
    directory: Path,
    issues: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in manifest:
        doc_id = str(item.get("doc_id") or "")
        path = _resolve_artifact(directory, item.get("chunked_path"), f"{doc_id}.chunks.json")
        data = _read_json(path, issues, "chunk_json_invalid", doc_id, item.get("source_name"))
        if data is None:
            continue
        metadata = data.get("metadata") or {}
        chunks = data.get("chunks") or []
        artifact_doc_id = str(metadata.get("doc_id") or "")
        artifact_source_name = str(metadata.get("source_name") or "")
        if doc_id and artifact_doc_id and doc_id != artifact_doc_id:
            _issue(
                issues,
                "error",
                "chunks_manifest_doc_id_mismatch",
                f"Manifest doc_id differs from chunk JSON: {path.name}",
                doc_id=artifact_doc_id,
                source_name=artifact_source_name or item.get("source_name"),
                details={"manifest_doc_id": doc_id, "artifact_doc_id": artifact_doc_id},
            )
        manifest_source_name = str(item.get("source_name") or "")
        if manifest_source_name and artifact_source_name and manifest_source_name != artifact_source_name:
            _issue(
                issues,
                "error",
                "chunks_manifest_source_mismatch",
                f"Manifest source_name differs from chunk JSON: {path.name}",
                doc_id=artifact_doc_id or doc_id,
                source_name=artifact_source_name,
                details={
                    "manifest_source_name": manifest_source_name,
                    "artifact_source_name": artifact_source_name,
                },
            )
        if data.get("chunk_count") != len(chunks) or item.get("chunks") != len(chunks):
            _issue(
                issues,
                "error",
                "chunk_count_mismatch",
                f"Chunk counts disagree with actual data: {path.name}",
                doc_id=metadata.get("doc_id") or doc_id,
                source_name=metadata.get("source_name") or item.get("source_name"),
                details={
                    "file_chunk_count": data.get("chunk_count"),
                    "manifest_chunks": item.get("chunks"),
                    "actual_chunks": len(chunks),
                },
            )
        records.append(
            {
                "doc_id": metadata.get("doc_id") or doc_id,
                "source_name": metadata.get("source_name") or item.get("source_name"),
                "path": path,
                "data": data,
                "max_chars": item.get("max_chars"),
            }
        )
    return records


def _read_json(
    path: Path,
    issues: list[dict[str, Any]],
    code: str,
    doc_id: Any,
    source_name: Any,
) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _issue(
            issues,
            "error",
            code,
            f"Cannot read artifact {path}: {exc}",
            doc_id=doc_id,
            source_name=source_name,
        )
        return None
    if not isinstance(data, dict):
        _issue(
            issues,
            "error",
            code,
            f"Artifact must contain a JSON object: {path}",
            doc_id=doc_id,
            source_name=source_name,
        )
        return None
    return data


def _audit_document(
    source_file: Path,
    extracted_record: dict[str, Any],
    chunk_records: list[dict[str, Any]],
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    data = extracted_record["data"]
    metadata = data.get("metadata") or {}
    blocks = data.get("blocks") or []
    doc_id = str(metadata.get("doc_id") or extracted_record.get("doc_id") or "")
    source_name = source_file.name
    report: dict[str, Any] = {
        "doc_id": doc_id,
        "source_name": source_name,
        "source_path": str(source_file),
        "blocks": len(blocks),
        "chunks": 0,
    }
    if not doc_id:
        _issue(
            issues,
            "error",
            "extraction_doc_id_missing",
            "Extraction metadata has no doc_id",
            source_name=source_name,
        )
    if not blocks:
        _issue(
            issues,
            "error",
            "extraction_blocks_empty",
            "Extraction contains no document blocks",
            doc_id=doc_id,
            source_name=source_name,
        )

    fresh_data: dict[str, Any] | None = None
    try:
        fresh_data = extract_docx(source_file).to_dict()
        source_text = _structured_document_text(fresh_data)
    except Exception as exc:
        _issue(
            issues,
            "error",
            "source_docx_unreadable",
            f"Cannot read DOCX: {exc}",
            doc_id=doc_id,
            source_name=source_name,
        )
        source_text = ""
    if fresh_data is not None:
        report.update(
            _audit_extraction_snapshot(
                fresh_data,
                data,
                issues,
                doc_id=doc_id,
                source_name=source_name,
            )
        )
    extracted_text = "\n".join(str(block.get("text") or "") for block in blocks)
    structured_text = _structured_document_text(data)
    source_coverage = _vocabulary_coverage(source_text, structured_text)
    report["source_chars"] = len(source_text)
    report["extracted_chars"] = len(extracted_text)
    report["source_vocabulary_coverage"] = round(source_coverage, 4)
    if source_text and blocks and source_coverage < 0.7:
        _issue(
            issues,
            "error",
            "extraction_vocabulary_coverage_low",
            f"Parser retained only {source_coverage:.1%} of source vocabulary",
            doc_id=doc_id,
            source_name=source_name,
            details={"missing_terms": _missing_vocabulary(source_text, structured_text)},
        )
    elif source_text and blocks and source_coverage < 0.9:
        _issue(
            issues,
            "warning",
            "extraction_vocabulary_coverage_partial",
            f"Parser retained {source_coverage:.1%} of source vocabulary",
            doc_id=doc_id,
            source_name=source_name,
            details={"missing_terms": _missing_vocabulary(source_text, structured_text)},
        )

    if len(chunk_records) != 1:
        if not chunk_records:
            _issue(
                issues,
                "error",
                "extracted_document_not_chunked",
                "Extracted document has no chunk artifact",
                doc_id=doc_id,
                source_name=source_name,
            )
        else:
            _issue(
                issues,
                "error",
                "duplicate_chunk_document",
                "Multiple chunk artifacts use the same doc_id",
                doc_id=doc_id,
                source_name=source_name,
            )
        return report

    chunk_record = chunk_records[0]
    chunks = chunk_record["data"].get("chunks") or []
    max_chars = _chunk_max_chars(chunk_record.get("max_chars"), issues, doc_id, source_name)
    report["chunk_max_chars"] = max_chars
    report.update(
        _audit_chunk_snapshot(
            data,
            chunk_record["data"],
            max_chars,
            issues,
            doc_id=doc_id,
            source_name=source_name,
        )
    )
    report["chunks"] = len(chunks)
    metadata_chunks = [chunk for chunk in chunks if chunk.get("chunk_type") == "metadata"]
    report["metadata_chunks"] = len(metadata_chunks)
    if not metadata_chunks:
        _issue(
            issues,
            "error",
            "metadata_chunk_count_invalid",
            "Expected at least one metadata chunk, found none",
            doc_id=doc_id,
            source_name=source_name,
        )
    indexable_blocks = [block for block in blocks if block.get("text")]
    excluded_blocks: list[dict[str, Any]] = []
    report["intentionally_non_indexed_blocks"] = len(excluded_blocks)
    report["intentionally_non_indexed_chars"] = sum(
        len(str(block.get("text") or "")) for block in excluded_blocks
    )
    if not indexable_blocks:
        _issue(
            issues,
            "error",
            "indexable_blocks_missing",
            "Extraction contains no substantive blocks eligible for indexing",
            doc_id=doc_id,
            source_name=source_name,
        )
    invalid_block_ids = [
        index for index, block in enumerate(indexable_blocks) if not block.get("block_id")
    ]
    if invalid_block_ids:
        _issue(
            issues,
            "error",
            "extraction_block_id_missing",
            f"{len(invalid_block_ids)} indexable extraction blocks have no block_id",
            doc_id=doc_id,
            source_name=source_name,
            details={"block_indexes": invalid_block_ids[:50]},
        )
    expected_block_ids = {
        str(block["block_id"]) for block in indexable_blocks if block.get("block_id")
    }
    covered_block_ids: set[str] = set()
    chunk_texts: list[str] = []
    searchable_texts: list[str] = []
    seen_chunk_ids: set[str] = set()
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id") or "")
        if not chunk_id:
            _issue(
                issues,
                "error",
                "chunk_id_missing",
                "Chunk has no chunk_id",
                doc_id=doc_id,
                source_name=source_name,
            )
        elif chunk_id in seen_chunk_ids:
            _issue(
                issues,
                "error",
                "duplicate_chunk_id",
                f"Duplicate chunk_id: {chunk_id}",
                doc_id=doc_id,
                source_name=source_name,
            )
        seen_chunk_ids.add(chunk_id)
        raw_text = str(chunk.get("raw_text") or "")
        searchable_text = str(chunk.get("searchable_text") or "")
        block_ids = chunk.get("block_ids") or []
        if not block_ids:
            _issue(
                issues,
                "error",
                "chunk_block_ids_missing",
                f"Chunk has no block_ids: {chunk_id or '<missing id>'}",
                doc_id=doc_id,
                source_name=source_name,
            )
        if not raw_text or not searchable_text:
            _issue(
                issues,
                "error",
                "empty_chunk_text",
                f"Chunk has empty text: {chunk_id or '<missing id>'}",
                doc_id=doc_id,
                source_name=source_name,
            )
        if int(chunk.get("char_count") or 0) != len(raw_text):
            _issue(
                issues,
                "error",
                "chunk_char_count_mismatch",
                f"char_count differs from raw_text length: {chunk_id}",
                doc_id=doc_id,
                source_name=source_name,
            )
        covered_block_ids.update(str(value) for value in block_ids)
        searchable_texts.append(searchable_text)
        if chunk.get("chunk_type") != "metadata":
            chunk_texts.append(raw_text)

    missing_block_ids = sorted(expected_block_ids - covered_block_ids)
    unknown_block_ids = sorted(covered_block_ids - expected_block_ids - {"metadata"})
    if missing_block_ids:
        _issue(
            issues,
            "error",
            "chunk_blocks_missing",
            f"{len(missing_block_ids)} extracted blocks are absent from chunks",
            doc_id=doc_id,
            source_name=source_name,
            details={"block_ids": missing_block_ids},
        )
    if unknown_block_ids:
        _issue(
            issues,
            "error",
            "chunk_blocks_unknown",
            f"{len(unknown_block_ids)} chunk block_ids are absent from extraction",
            doc_id=doc_id,
            source_name=source_name,
            details={"block_ids": unknown_block_ids},
        )

    expected_text = "\n".join(str(block.get("text") or "") for block in indexable_blocks)
    actual_text = "\n".join(chunk_texts)
    text_coverage = _character_coverage(expected_text, actual_text)
    report["indexable_blocks"] = len(indexable_blocks)
    report["covered_blocks"] = len(expected_block_ids & covered_block_ids)
    report["chunk_text_coverage"] = round(text_coverage, 4)
    report["chunk_chars"] = len(actual_text)
    searchable_corpus = "\n".join(searchable_texts)
    unsearchable_blocks = [
        {
            "block_id": str(block.get("block_id") or ""),
            "kind": str(block.get("kind") or ""),
            "text": str(block.get("text") or "")[:240],
            "coverage": round(
                _token_coverage(str(block.get("text") or ""), searchable_corpus),
                4,
            ),
        }
        for block in excluded_blocks
        if _token_coverage(str(block.get("text") or ""), searchable_corpus) < 0.95
    ]
    report["unsearchable_non_indexed_blocks"] = len(unsearchable_blocks)
    if unsearchable_blocks:
        _issue(
            issues,
            "error",
            "non_indexed_blocks_not_searchable",
            f"{len(unsearchable_blocks)} excluded heading or front-matter blocks are absent from searchable text",
            doc_id=doc_id,
            source_name=source_name,
            details={"blocks": unsearchable_blocks[:50]},
        )
    if expected_text and text_coverage < 0.98:
        _issue(
            issues,
            "error",
            "chunk_text_coverage_low",
            f"Only {text_coverage:.1%} of extracted content is represented in chunks",
            doc_id=doc_id,
            source_name=source_name,
        )
    return report


def _audit_chunk_snapshot(
    extracted_data: dict[str, Any],
    stored_data: dict[str, Any],
    max_chars: int,
    issues: list[dict[str, Any]],
    *,
    doc_id: str,
    source_name: str,
) -> dict[str, Any]:
    fresh_data = chunk_document(extracted_data, max_chars=max_chars)
    if fresh_data.get("metadata") != stored_data.get("metadata"):
        _issue(
            issues,
            "error",
            "chunk_metadata_stale",
            "Chunk artifact metadata differs from the extraction artifact",
            doc_id=doc_id,
            source_name=source_name,
        )

    fresh_chunks = fresh_data.get("chunks") or []
    stored_chunks = stored_data.get("chunks") or []
    fresh_by_id = _chunks_by_id(fresh_chunks)
    stored_by_id = _chunks_by_id(stored_chunks)
    missing_ids = sorted(set(fresh_by_id) - set(stored_by_id))
    stale_ids = sorted(set(stored_by_id) - set(fresh_by_id))
    changed_ids = sorted(
        chunk_id
        for chunk_id in set(fresh_by_id) & set(stored_by_id)
        if len(fresh_by_id[chunk_id]) == 1
        and len(stored_by_id[chunk_id]) == 1
        and fresh_by_id[chunk_id][0] != stored_by_id[chunk_id][0]
    )
    if missing_ids:
        _issue(
            issues,
            "error",
            "chunk_snapshot_missing",
            f"Chunk artifact is missing {len(missing_ids)} chunks produced by the current chunker",
            doc_id=doc_id,
            source_name=source_name,
            details={"chunk_ids": missing_ids[:50]},
        )
    if stale_ids:
        _issue(
            issues,
            "error",
            "chunk_snapshot_stale",
            f"Chunk artifact contains {len(stale_ids)} chunks no longer produced by the current chunker",
            doc_id=doc_id,
            source_name=source_name,
            details={"chunk_ids": stale_ids[:50]},
        )
    if changed_ids:
        changed_fields = {
            chunk_id: _changed_fields(
                fresh_by_id[chunk_id][0],
                stored_by_id[chunk_id][0],
            )
            for chunk_id in changed_ids[:50]
        }
        _issue(
            issues,
            "error",
            "chunk_snapshot_changed",
            f"{len(changed_ids)} chunks differ from the current deterministic chunker output",
            doc_id=doc_id,
            source_name=source_name,
            details={"chunks": changed_fields},
        )
    return {
        "current_chunker_chunks": len(fresh_chunks),
        "chunk_snapshot_missing": len(missing_ids),
        "chunk_snapshot_stale": len(stale_ids),
        "chunk_snapshot_changed": len(changed_ids),
    }


def _audit_extraction_snapshot(
    fresh_data: dict[str, Any],
    stored_data: dict[str, Any],
    issues: list[dict[str, Any]],
    *,
    doc_id: str,
    source_name: str,
) -> dict[str, Any]:
    fresh_metadata = fresh_data.get("metadata") or {}
    stored_metadata = stored_data.get("metadata") or {}
    fresh_doc_id = str(fresh_metadata.get("doc_id") or "")
    if fresh_doc_id != doc_id:
        _issue(
            issues,
            "error",
            "extraction_doc_id_stale",
            f"Current extractor produces doc_id {fresh_doc_id!r}, artifact contains {doc_id!r}",
            doc_id=doc_id,
            source_name=source_name,
        )

    metadata_fields = (
        "source_name",
        "index_code",
        "display_title",
        "document_kind",
        "version",
        "approval_date",
        "approval_order_number",
        "effective_date",
        "declared_pages",
        "title_lines",
    )
    mismatched_metadata = [
        field
        for field in metadata_fields
        if fresh_metadata.get(field) != stored_metadata.get(field)
    ]
    if mismatched_metadata:
        _issue(
            issues,
            "error",
            "extraction_metadata_stale",
            "Stored extraction metadata differs from the current DOCX",
            doc_id=doc_id,
            source_name=source_name,
            details={"fields": mismatched_metadata},
        )

    fresh_blocks = fresh_data.get("blocks") or []
    stored_blocks = stored_data.get("blocks") or []
    fresh_by_id = _blocks_by_id(fresh_blocks)
    stored_by_id = _blocks_by_id(stored_blocks)
    duplicate_fresh = sorted(block_id for block_id, values in fresh_by_id.items() if len(values) > 1)
    duplicate_stored = sorted(block_id for block_id, values in stored_by_id.items() if len(values) > 1)
    if duplicate_fresh:
        _issue(
            issues,
            "error",
            "extractor_duplicate_block_id",
            f"Current extractor produces {len(duplicate_fresh)} duplicate block IDs",
            doc_id=doc_id,
            source_name=source_name,
            details={"block_ids": duplicate_fresh[:50]},
        )
    if duplicate_stored:
        _issue(
            issues,
            "error",
            "extraction_duplicate_block_id",
            f"Stored extraction contains {len(duplicate_stored)} duplicate block IDs",
            doc_id=doc_id,
            source_name=source_name,
            details={"block_ids": duplicate_stored[:50]},
        )

    missing_ids = sorted(set(fresh_by_id) - set(stored_by_id))
    stale_ids = sorted(set(stored_by_id) - set(fresh_by_id))
    changed_ids = sorted(
        block_id
        for block_id in set(fresh_by_id) & set(stored_by_id)
        if len(fresh_by_id[block_id]) == 1
        and len(stored_by_id[block_id]) == 1
        and _block_signature(fresh_by_id[block_id][0])
        != _block_signature(stored_by_id[block_id][0])
    )
    if missing_ids:
        _issue(
            issues,
            "error",
            "extraction_blocks_missing",
            f"Extraction artifact is missing {len(missing_ids)} blocks from the current DOCX",
            doc_id=doc_id,
            source_name=source_name,
            details={"block_ids": missing_ids[:50]},
        )
    if stale_ids:
        _issue(
            issues,
            "error",
            "extraction_blocks_stale",
            f"Extraction artifact contains {len(stale_ids)} blocks absent from the current DOCX",
            doc_id=doc_id,
            source_name=source_name,
            details={"block_ids": stale_ids[:50]},
        )
    if changed_ids:
        _issue(
            issues,
            "error",
            "extraction_blocks_changed",
            f"Text or structure changed in {len(changed_ids)} extracted blocks",
            doc_id=doc_id,
            source_name=source_name,
            details={"block_ids": changed_ids[:50]},
        )
    return {
        "current_extractor_blocks": len(fresh_blocks),
        "extraction_missing_blocks": len(missing_ids),
        "extraction_stale_blocks": len(stale_ids),
        "extraction_changed_blocks": len(changed_ids),
    }


def _blocks_by_id(blocks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for block in blocks:
        grouped[str(block.get("block_id") or "")].append(block)
    return dict(grouped)


def _chunks_by_id(chunks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        grouped[str(chunk.get("chunk_id") or "")].append(chunk)
    return dict(grouped)


def _changed_fields(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    return sorted(
        field
        for field in set(expected) | set(actual)
        if expected.get(field) != actual.get(field)
    )


def _chunk_max_chars(
    value: Any,
    issues: list[dict[str, Any]],
    doc_id: str,
    source_name: str,
) -> int:
    if value in (None, ""):
        return DEFAULT_MAX_CHARS
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = 0
    if result > 0:
        return result
    _issue(
        issues,
        "error",
        "chunk_max_chars_invalid",
        f"Chunk manifest contains invalid max_chars: {value!r}",
        doc_id=doc_id,
        source_name=source_name,
    )
    return DEFAULT_MAX_CHARS


def _block_signature(block: dict[str, Any]) -> tuple[Any, ...]:
    return (
        block.get("kind"),
        str(block.get("text") or ""),
        block.get("source_kind"),
        block.get("section_number"),
        block.get("section_title"),
        block.get("subsection_number"),
        block.get("subsection_title"),
        block.get("item_number"),
        block.get("item_marker"),
        block.get("appendix_number"),
        block.get("appendix_title"),
        block.get("heading_level"),
        tuple(block.get("section_path") or []),
    )


def _structured_document_text(data: dict[str, Any]) -> str:
    values: list[str] = [
        "индекс нд индекс лна документ тип организация версия дата утверждения",
        "номер приказа дата ввода в действие введено в действие листов файл",
        "утверждено утвержден приказом приложение раздел пункт",
    ]

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for nested in value.values():
                collect(nested)
        elif isinstance(value, list):
            for nested in value:
                collect(nested)
        elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
            text = str(value).strip()
            if text:
                values.append(text)

    collect(data.get("metadata") or {})
    collect(data.get("blocks") or [])
    return "\n".join(values)


def _missing_vocabulary(expected: str, actual: str, limit: int = 50) -> list[str]:
    return sorted(set(_tokens(expected)) - set(_tokens(actual)))[:limit]


def _audit_database(
    value: str | None,
    expected_documents: dict[str, dict[str, Any]],
    expected_chunks: dict[str, dict[str, Any]],
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    url = database_url(value)
    if not url:
        _issue(
            issues,
            "error",
            "database_url_missing",
            "DATABASE_URL is not configured",
        )
        return {"checked": False, "error": "DATABASE_URL is not configured"}
    try:
        with connect(url) as conn:
            documents = list(
                conn.execute(
                    "SELECT doc_id, source_name, metadata FROM doc_documents ORDER BY doc_id"
                ).fetchall()
            )
            chunks = list(
                conn.execute(
                    """
                    SELECT chunk_id, doc_id, source_name, raw_text, searchable_text,
                           block_ids, char_count, embedding_model, metadata,
                           embedding IS NULL AS embedding_missing,
                           vector_dims(embedding) AS embedding_dim
                    FROM doc_chunks
                    ORDER BY chunk_id
                    """
                ).fetchall()
            )
            indexes = [
                str(row["indexname"])
                for row in conn.execute(
                    "SELECT indexname FROM pg_indexes WHERE tablename = 'doc_chunks' ORDER BY indexname"
                ).fetchall()
            ]
    except Exception as exc:
        _issue(
            issues,
            "error",
            "database_audit_failed",
            f"Cannot audit PostgreSQL: {exc}",
        )
        return {"checked": False, "url": redact_url(url), "error": str(exc)}

    db_documents = {str(row["doc_id"]): row for row in documents}
    db_chunks = {str(row["chunk_id"]): row for row in chunks}
    missing_documents = sorted(set(expected_documents) - set(db_documents))
    stale_documents = sorted(set(db_documents) - set(expected_documents))
    missing_chunks = sorted(set(expected_chunks) - set(db_chunks))
    stale_chunks = sorted(set(db_chunks) - set(expected_chunks))

    if missing_documents:
        _issue(
            issues,
            "error",
            "database_documents_missing",
            f"PostgreSQL is missing {len(missing_documents)} documents",
            details={"doc_ids": missing_documents},
        )
    if stale_documents:
        _issue(
            issues,
            "error",
            "database_documents_stale",
            f"PostgreSQL contains {len(stale_documents)} stale documents",
            details={"doc_ids": stale_documents},
        )
    if missing_chunks:
        _issue(
            issues,
            "error",
            "database_chunks_missing",
            f"PostgreSQL is missing {len(missing_chunks)} artifact chunks",
            details={"sample_chunk_ids": missing_chunks[:50]},
        )
    if stale_chunks:
        _issue(
            issues,
            "error",
            "database_chunks_stale",
            f"PostgreSQL contains {len(stale_chunks)} stale chunks",
            details={"sample_chunk_ids": stale_chunks[:50]},
        )

    missing_embeddings = sorted(
        str(row["chunk_id"]) for row in chunks if row.get("embedding_missing")
    )
    if missing_embeddings:
        _issue(
            issues,
            "error",
            "database_embeddings_missing",
            f"PostgreSQL contains {len(missing_embeddings)} chunks without embeddings",
            details={"sample_chunk_ids": missing_embeddings[:50]},
        )

    mismatched_documents: list[str] = []
    for doc_id in sorted(set(expected_documents) & set(db_documents)):
        expected = expected_documents[doc_id]
        actual = db_documents[doc_id]
        if (
            str(expected.get("source_name") or "") != str(actual.get("source_name") or "")
            or expected.get("metadata") != actual.get("metadata")
        ):
            mismatched_documents.append(doc_id)
    if mismatched_documents:
        _issue(
            issues,
            "error",
            "database_document_content_mismatch",
            f"{len(mismatched_documents)} PostgreSQL documents differ from artifacts",
            details={"doc_ids": mismatched_documents[:50]},
        )

    mismatched_chunks: list[str] = []
    for chunk_id in sorted(set(expected_chunks) & set(db_chunks)):
        expected = expected_chunks[chunk_id]
        actual = db_chunks[chunk_id]
        if (
            str(expected.get("raw_text") or "") != str(actual.get("raw_text") or "")
            or str(expected.get("searchable_text") or "") != str(actual.get("searchable_text") or "")
            or int(expected.get("char_count") or 0) != int(actual.get("char_count") or 0)
            or list(expected.get("block_ids") or []) != list(actual.get("block_ids") or [])
            or expected != actual.get("metadata")
        ):
            mismatched_chunks.append(chunk_id)
    if mismatched_chunks:
        _issue(
            issues,
            "error",
            "database_chunk_content_mismatch",
            f"{len(mismatched_chunks)} PostgreSQL chunks differ from artifacts",
            details={"sample_chunk_ids": mismatched_chunks[:50]},
        )

    if not any("embedding_hnsw" in name or "embedding_ivfflat" in name for name in indexes):
        _issue(
            issues,
            "warning",
            "database_vector_index_missing",
            "No HNSW or IVFFLAT index was found for embeddings",
        )
    embedding_models = sorted(
        {str(row.get("embedding_model") or "") for row in chunks if row.get("embedding_model")}
    )
    embedding_dimensions = sorted(
        {int(row["embedding_dim"]) for row in chunks if row.get("embedding_dim") is not None}
    )
    if chunks and not embedding_models:
        _issue(
            issues,
            "error",
            "database_embedding_model_missing",
            "Indexed chunks do not identify their embedding model",
        )
    if len(embedding_models) > 1:
        _issue(
            issues,
            "warning",
            "multiple_embedding_models",
            "Chunks were indexed with multiple embedding models",
            details={"models": embedding_models},
        )
    if len(embedding_dimensions) > 1:
        _issue(
            issues,
            "error",
            "multiple_embedding_dimensions",
            "Chunks use multiple embedding dimensions",
            details={"dimensions": embedding_dimensions},
        )
    return {
        "checked": True,
        "url": redact_url(url),
        "documents": len(documents),
        "chunks": len(chunks),
        "embedding_models": embedding_models,
        "embedding_dimensions": embedding_dimensions,
        "indexes": indexes,
        "missing_documents": len(missing_documents),
        "stale_documents": len(stale_documents),
        "missing_chunks": len(missing_chunks),
        "stale_chunks": len(stale_chunks),
        "mismatched_chunks": len(mismatched_chunks),
        "mismatched_documents": len(mismatched_documents),
        "missing_embeddings": len(missing_embeddings),
    }


def _group_by_name(paths: list[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        grouped[_key(path.name)].append(path)
    return dict(grouped)


def _group_records(
    records: list[dict[str, Any]],
    field: str,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_key(record.get(field))].append(record)
    return dict(grouped)


def _key(value: Any) -> str:
    return str(value or "").strip().casefold()


def _tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in WORD_RE.finditer(text)]


def _vocabulary_coverage(expected: str, actual: str) -> float:
    expected_tokens = set(_tokens(expected))
    if not expected_tokens:
        return 1.0
    return len(expected_tokens & set(_tokens(actual))) / len(expected_tokens)


def _token_coverage(expected: str, actual: str) -> float:
    expected_counts = Counter(_tokens(expected))
    if not expected_counts:
        return 1.0
    actual_counts = Counter(_tokens(actual))
    matched = sum(min(count, actual_counts[token]) for token, count in expected_counts.items())
    return matched / sum(expected_counts.values())


def _character_coverage(expected: str, actual: str) -> float:
    expected_counts = Counter(character.casefold() for character in expected if character.isalnum())
    if not expected_counts:
        return 1.0
    actual_counts = Counter(character.casefold() for character in actual if character.isalnum())
    matched = sum(min(count, actual_counts[character]) for character, count in expected_counts.items())
    return matched / sum(expected_counts.values())


def _issue(
    issues: list[dict[str, Any]],
    level: str,
    code: str,
    message: str,
    *,
    doc_id: Any = None,
    source_name: Any = None,
    details: dict[str, Any] | None = None,
) -> None:
    item: dict[str, Any] = {"level": level, "code": code, "message": message}
    if doc_id:
        item["doc_id"] = str(doc_id)
    if source_name:
        item["source_name"] = str(source_name)
    if details:
        item["details"] = details
    issues.append(item)
