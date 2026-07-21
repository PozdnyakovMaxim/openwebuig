from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from typing import Any, Iterable


CANDIDATE_SCHEMA_VERSION = 1
CANDIDATE_TOOL_VERSION = "openwebuig-rebuild-corpus/1"
AUDIT_CERTIFICATION_TOOL_VERSION = "openwebuig-corpus-audit-certification/1"
HASH_ALGORITHM = "sha256"

_SOURCE_KIND = "source_docx"
_ARTIFACT_KINDS = {
    "extracted_json",
    "chunk_json",
    "extraction_manifest",
    "chunk_manifest",
}
_EXPECTED_KINDS = {_SOURCE_KIND, *_ARTIFACT_KINDS}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_integrity_manifest(
    *,
    docs_dir: Path,
    candidate_dir: Path,
    source_files: Iterable[Path],
    extracted_files: Iterable[Path],
    chunk_files: Iterable[Path],
) -> dict[str, Any]:
    docs_dir = docs_dir.resolve()
    candidate_dir = candidate_dir.resolve()
    files: list[dict[str, str]] = []

    for source_path in source_files:
        files.append(
            _file_record(
                _SOURCE_KIND,
                source_path.resolve(),
                relative_to=docs_dir,
            )
        )
    for path in extracted_files:
        files.append(
            _file_record("extracted_json", path.resolve(), relative_to=candidate_dir)
        )
    for path in chunk_files:
        files.append(_file_record("chunk_json", path.resolve(), relative_to=candidate_dir))
    files.append(
        _file_record(
            "extraction_manifest",
            candidate_dir / "extracted" / "manifest.json",
            relative_to=candidate_dir,
        )
    )
    files.append(
        _file_record(
            "chunk_manifest",
            candidate_dir / "chunks" / "manifest.json",
            relative_to=candidate_dir,
        )
    )
    files.sort(key=lambda item: (item["kind"], item["path"]))
    return {
        "algorithm": HASH_ALGORITHM,
        "source_root": str(docs_dir),
        "files": files,
    }


def validate_integrity_metadata(marker: dict[str, Any]) -> None:
    if not isinstance(marker, dict):
        raise ValueError("Candidate marker must be a JSON object")
    schema_version = marker.get("schema_version")
    if schema_version != CANDIDATE_SCHEMA_VERSION:
        raise ValueError(
            "Candidate marker has unsupported schema_version: "
            f"{schema_version!r}; expected {CANDIDATE_SCHEMA_VERSION}"
        )
    tool_version = marker.get("tool_version")
    if not isinstance(tool_version, str) or not tool_version.strip():
        raise ValueError(f"Candidate marker has invalid tool_version: {tool_version!r}")

    integrity = marker.get("integrity")
    if not isinstance(integrity, dict):
        raise ValueError("Candidate marker is missing integrity metadata")
    if integrity.get("algorithm") != HASH_ALGORITHM:
        raise ValueError(
            f"Candidate marker has unsupported hash algorithm: {integrity.get('algorithm')!r}"
        )
    source_root = integrity.get("source_root")
    if not isinstance(source_root, str) or not source_root.strip():
        raise ValueError("Candidate marker integrity.source_root is empty")
    files = integrity.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Candidate marker integrity.files is empty")

    seen: set[tuple[str, str]] = set()
    kinds: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("Candidate marker contains a non-object integrity file entry")
        kind = str(entry.get("kind") or "").strip()
        raw_path = str(entry.get("path") or "").strip()
        checksum = str(entry.get("sha256") or "").strip().casefold()
        if kind not in _EXPECTED_KINDS:
            raise ValueError(f"Candidate marker has unsupported integrity file kind: {kind!r}")
        _validated_relative_path(raw_path)
        if len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum):
            raise ValueError(f"Candidate marker has invalid SHA-256 for {raw_path!r}")
        identity = (kind, raw_path)
        if identity in seen:
            raise ValueError(f"Candidate marker has duplicate integrity entry: {kind}:{raw_path}")
        seen.add(identity)
        kinds.add(kind)

    missing_kinds = sorted(_EXPECTED_KINDS - kinds)
    if missing_kinds:
        raise ValueError(
            "Candidate marker is missing integrity file kinds: " + ", ".join(missing_kinds)
        )


def verify_candidate_integrity(candidate_dir: Path, marker: dict[str, Any]) -> None:
    validate_integrity_metadata(marker)
    candidate_dir = candidate_dir.resolve()
    integrity = marker["integrity"]
    source_root = Path(integrity["source_root"]).expanduser().resolve()

    counts: dict[str, int] = {}
    source_hashes: dict[str, str] = {}
    artifact_paths: list[tuple[str, Path]] = []
    for entry in integrity["files"]:
        kind = str(entry["kind"])
        relative_path = _validated_relative_path(str(entry["path"]))
        root = source_root if kind == _SOURCE_KIND else candidate_dir
        path = (root / relative_path).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Candidate integrity path escapes its root: {relative_path}")
        if not path.is_file():
            raise ValueError(f"Candidate integrity file is missing: {path}")
        actual = sha256_file(path)
        expected = str(entry["sha256"]).casefold()
        if not hmac.compare_digest(actual, expected):
            raise ValueError(
                f"Candidate integrity check failed for {kind} {relative_path}: "
                f"expected {expected}, received {actual}"
            )
        counts[kind] = counts.get(kind, 0) + 1
        if kind == _SOURCE_KIND:
            source_hashes[relative_path.as_posix()] = expected
        elif kind in {"extracted_json", "chunk_json"}:
            artifact_paths.append((kind, path))

    expected_documents = marker.get("documents")
    for kind in (_SOURCE_KIND, "extracted_json", "chunk_json"):
        if counts.get(kind) != expected_documents:
            raise ValueError(
                f"Candidate integrity count mismatch for {kind}: "
                f"expected {expected_documents}, found {counts.get(kind, 0)}"
            )
    for kind in ("extraction_manifest", "chunk_manifest"):
        if counts.get(kind) != 1:
            raise ValueError(
                f"Candidate integrity count mismatch for {kind}: expected 1, "
                f"found {counts.get(kind, 0)}"
            )

    discovered_sources = {
        path.resolve().relative_to(source_root).as_posix()
        for path in source_root.rglob("*")
        if path.is_file()
        and path.suffix.casefold() == ".docx"
        and not path.name.startswith("~$")
        and path.resolve().is_relative_to(source_root)
    }
    if discovered_sources != set(source_hashes):
        missing = sorted(set(source_hashes) - discovered_sources)
        unexpected = sorted(discovered_sources - set(source_hashes))
        raise ValueError(
            "Candidate source inventory changed since discovery: "
            f"missing={missing}, unexpected={unexpected}"
        )

    bindings: dict[str, dict[str, tuple[str, str]]] = {}
    for kind, artifact_path in artifact_paths:
        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Candidate artifact is not valid JSON: {artifact_path}: {exc}") from exc
        metadata = payload.get("metadata") if isinstance(payload, dict) else None
        if not isinstance(metadata, dict):
            raise ValueError(f"Candidate artifact has no metadata object: {artifact_path}")
        doc_id = str(metadata.get("doc_id") or "").strip()
        stored_source_path = str(metadata.get("source_path") or "").strip()
        stored_source_hash = str(metadata.get("source_sha256") or "").strip().casefold()
        if not doc_id or not stored_source_path or not stored_source_hash:
            raise ValueError(
                f"Candidate artifact has incomplete source binding: {artifact_path}"
            )
        resolved_source_path = Path(stored_source_path).expanduser().resolve()
        try:
            source_relative = resolved_source_path.relative_to(source_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"Candidate artifact source path is outside source_root: {stored_source_path}"
            ) from exc
        manifest_hash = source_hashes.get(source_relative)
        if manifest_hash is None:
            raise ValueError(
                f"Candidate artifact references an untracked source: {source_relative}"
            )
        if not hmac.compare_digest(stored_source_hash, manifest_hash):
            raise ValueError(
                f"Candidate artifact source hash mismatch for {doc_id}: "
                f"metadata={stored_source_hash}, manifest={manifest_hash}"
            )
        source_name = str(metadata.get("source_name") or "").strip()
        if source_name != resolved_source_path.name:
            raise ValueError(
                f"Candidate artifact source_name mismatch for {doc_id}: "
                f"metadata={source_name!r}, path={resolved_source_path.name!r}"
            )
        document_bindings = bindings.setdefault(doc_id, {})
        if kind in document_bindings:
            raise ValueError(f"Candidate has duplicate {kind} artifacts for document {doc_id}")
        document_bindings[kind] = (source_relative, stored_source_hash)

    if len(bindings) != expected_documents:
        raise ValueError(
            "Candidate artifact document count mismatch: "
            f"expected {expected_documents}, found {len(bindings)}"
        )
    bound_sources: set[str] = set()
    for doc_id, values in bindings.items():
        if set(values) != {"extracted_json", "chunk_json"}:
            raise ValueError(f"Candidate artifacts are incomplete for document {doc_id}")
        if values["extracted_json"] != values["chunk_json"]:
            raise ValueError(
                f"Extracted and chunk artifacts reference different sources for {doc_id}"
            )
        source_relative = values["extracted_json"][0]
        if source_relative in bound_sources:
            raise ValueError(
                f"Multiple candidate documents reference the same source: {source_relative}"
            )
        bound_sources.add(source_relative)
    if bound_sources != set(source_hashes):
        raise ValueError("Candidate artifacts do not bind every source document exactly once")


def certify_candidate_audit(candidate_dir: Path, report_path: Path) -> dict[str, Any]:
    candidate_dir = candidate_dir.resolve()
    report_path = report_path.resolve()
    ready_path = candidate_dir / "READY"
    if not ready_path.is_file():
        raise ValueError(f"Cannot certify candidate without READY: {ready_path}")
    if not report_path.is_file() or not report_path.is_relative_to(candidate_dir):
        raise ValueError("Certified audit report must be a file inside the candidate directory")
    try:
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read candidate audit inputs: {exc}") from exc
    if not isinstance(ready, dict) or not isinstance(report, dict):
        raise ValueError("Candidate READY and audit report must be JSON objects")
    verify_candidate_integrity(candidate_dir, ready)
    _validate_strict_audit_report(candidate_dir, ready, report)
    marker = {
        "tool_version": AUDIT_CERTIFICATION_TOOL_VERSION,
        "ready_sha256": sha256_file(ready_path),
        "report_path": report_path.relative_to(candidate_dir).as_posix(),
        "report_sha256": sha256_file(report_path),
    }
    (candidate_dir / "AUDITED").write_text(
        json.dumps(marker, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return marker


def verify_candidate_audit(candidate_dir: Path, ready: dict[str, Any]) -> dict[str, Any]:
    candidate_dir = candidate_dir.resolve()
    ready_path = candidate_dir / "READY"
    audited_path = candidate_dir / "AUDITED"
    if not audited_path.is_file():
        raise ValueError(
            f"Strict audit certification is missing: {audited_path}. "
            "Run audit_rag_corpus.py with --strict --skip-database --certify-candidate."
        )
    try:
        marker = json.loads(audited_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid candidate audit certification: {exc}") from exc
    if not isinstance(marker, dict):
        raise ValueError("Candidate audit certification must be a JSON object")
    if marker.get("tool_version") != AUDIT_CERTIFICATION_TOOL_VERSION:
        raise ValueError("Candidate audit certification has an unsupported tool version")
    if not ready_path.is_file() or not hmac.compare_digest(
        str(marker.get("ready_sha256") or "").casefold(),
        sha256_file(ready_path),
    ):
        raise ValueError("Candidate audit certification does not match READY")
    raw_report_path = str(marker.get("report_path") or "")
    relative_report_path = _validated_relative_path(raw_report_path)
    report_path = (candidate_dir / relative_report_path).resolve()
    if not report_path.is_relative_to(candidate_dir) or not report_path.is_file():
        raise ValueError("Certified candidate audit report is missing or outside the candidate")
    expected_report_hash = str(marker.get("report_sha256") or "").casefold()
    if not hmac.compare_digest(expected_report_hash, sha256_file(report_path)):
        raise ValueError("Certified candidate audit report hash mismatch")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Certified candidate audit report is invalid: {exc}") from exc
    _validate_strict_audit_report(candidate_dir, ready, report)
    return marker


def _validate_strict_audit_report(
    candidate_dir: Path,
    ready: dict[str, Any],
    report: dict[str, Any],
) -> None:
    if not isinstance(report, dict):
        raise ValueError("Candidate audit report must be a JSON object")
    summary = report.get("summary")
    if (
        report.get("status") != "ok"
        or not isinstance(summary, dict)
        or summary.get("errors") != 0
        or summary.get("warnings") != 0
    ):
        raise ValueError("Candidate audit report is not a clean strict audit")
    if summary.get("source_documents") != ready.get("documents"):
        raise ValueError("Candidate audit document count does not match READY")
    if summary.get("artifact_chunks") != ready.get("chunks"):
        raise ValueError("Candidate audit chunk count does not match READY")
    candidate = report.get("candidate")
    if not isinstance(candidate, dict):
        raise ValueError("Candidate audit report is not bound to READY")
    reported_candidate_dir = Path(
        str(candidate.get("candidate_dir") or "")
    ).expanduser().resolve()
    if reported_candidate_dir != candidate_dir:
        raise ValueError("Candidate audit report belongs to a different candidate directory")
    reported_ready_hash = str(candidate.get("ready_sha256") or "").casefold()
    actual_ready_hash = sha256_file(candidate_dir / "READY")
    if not hmac.compare_digest(reported_ready_hash, actual_ready_hash):
        raise ValueError("Candidate audit report READY digest does not match READY")
    paths = report.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("Candidate audit report has no paths")
    expected_paths = {
        "docs": Path(str(ready["integrity"]["source_root"])).expanduser().resolve(),
        "extracted": (candidate_dir / "extracted").resolve(),
        "chunks": (candidate_dir / "chunks").resolve(),
    }
    for name, expected in expected_paths.items():
        actual = Path(str(paths.get(name) or "")).expanduser().resolve()
        if actual != expected:
            raise ValueError(
                f"Candidate audit {name} path mismatch: expected {expected}, found {actual}"
            )


def _file_record(kind: str, path: Path, *, relative_to: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"Cannot hash missing candidate file: {path}")
    try:
        relative_path = path.relative_to(relative_to).as_posix()
    except ValueError as exc:
        raise ValueError(f"Candidate file is outside its expected root: {path}") from exc
    return {
        "kind": kind,
        "path": relative_path,
        "sha256": sha256_file(path),
    }


def _validated_relative_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not raw_path or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Candidate integrity path must be relative and contained: {raw_path!r}")
    return path
