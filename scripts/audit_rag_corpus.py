from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from document_search.corpus_audit import audit_corpus
from document_search.settings import load_env_file
from corpus_candidate_integrity import certify_candidate_audit, sha256_file


def _default_path(relative: str) -> Path:
    candidates = [
        ROOT.parent / "rag_template" / relative,
        Path.home() / "rag_template" / relative,
        ROOT / relative,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit source DOCX files, extraction artifacts, chunks, and pgvector."
    )
    parser.add_argument("--docs-dir", default=str(_default_path("docs")))
    parser.add_argument("--extracted-dir", default=str(_default_path("artifacts/extracted")))
    parser.add_argument("--chunks-dir", default=str(_default_path("artifacts/chunks")))
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--skip-database", action="store_true")
    parser.add_argument("--report", default=str(ROOT / "artifacts/rag_audit.json"))
    parser.add_argument("--strict", action="store_true", help="Return nonzero for warnings too.")
    parser.add_argument(
        "--certify-candidate",
        action="store_true",
        help="Write an AUDITED gate for a clean strict pre-database candidate audit.",
    )
    args = parser.parse_args()
    if args.certify_candidate and (not args.strict or not args.skip_database):
        parser.error("--certify-candidate requires --strict and --skip-database")

    candidate_dir: Path | None = None
    ready_path: Path | None = None
    ready_sha256_before: str | None = None
    if args.certify_candidate:
        extracted_parent = Path(args.extracted_dir).expanduser().resolve().parent
        chunks_parent = Path(args.chunks_dir).expanduser().resolve().parent
        if extracted_parent != chunks_parent:
            parser.error("Candidate extracted/ and chunks/ directories must share one parent")
        candidate_dir = extracted_parent
        ready_path = candidate_dir / "READY"
        if not ready_path.is_file():
            parser.error(f"Candidate READY is missing: {ready_path}")
        ready_sha256_before = sha256_file(ready_path)

    load_env_file(ROOT / ".env")
    load_env_file(ROOT.parent / "rag_template" / ".env")
    report = audit_corpus(
        args.docs_dir,
        args.extracted_dir,
        args.chunks_dir,
        database=args.database_url,
        skip_database=args.skip_database,
    )
    if candidate_dir is not None and ready_path is not None and ready_sha256_before is not None:
        ready_sha256_after = sha256_file(ready_path)
        if ready_sha256_after != ready_sha256_before:
            raise SystemExit("Candidate READY changed while the strict audit was running")
        report["candidate"] = {
            "candidate_dir": str(candidate_dir),
            "ready_sha256": ready_sha256_before,
        }
    output_path = Path(args.report).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = report["summary"]
    print(f"Status: {report['status']}")
    print(f"Source documents: {summary['source_documents']}")
    print(f"Extracted documents: {summary['extracted_documents']}")
    print(f"Chunked documents: {summary['chunked_documents']}")
    print(f"Artifact chunks: {summary['artifact_chunks']}")
    print(
        "Characters: "
        f"source={summary['source_characters']}, "
        f"extracted={summary['extracted_characters']}, "
        f"chunks={summary['chunk_characters']}"
    )
    print(
        "Independent OOXML inventory: "
        f"segments={summary['source_segments']}, "
        f"ignored={summary['source_ignored_segments']}, "
        f"tokens={summary['source_tokens']}, "
        f"missing={summary['source_missing_tokens']}, "
        f"coverage={summary['source_token_coverage']:.1%}"
    )
    print(
        "Blocks: "
        f"extracted={summary['extracted_blocks']}, "
        f"indexable={summary['indexable_blocks']}, "
        f"covered={summary['covered_blocks']}"
    )
    print(
        "Non-indexed blocks: "
        f"expected={summary['intentionally_non_indexed_blocks']}, "
        f"not searchable={summary['unsearchable_non_indexed_blocks']}"
    )
    database_report = report["database"]
    if database_report.get("checked"):
        print(f"PostgreSQL documents: {database_report['documents']}")
        print(f"PostgreSQL chunks: {database_report['chunks']}")
        print(f"PostgreSQL chunks without embeddings: {database_report['missing_embeddings']}")
    else:
        print("PostgreSQL: skipped" if args.skip_database else "PostgreSQL: audit failed")
    print(f"Errors: {summary['errors']}")
    print(f"Warnings: {summary['warnings']}")
    for issue in report["issues"]:
        label = issue["level"].upper()
        source = issue.get("source_name") or issue.get("doc_id") or "corpus"
        print(f"{label} [{issue['code']}] {source}: {issue['message']}")
    print(f"Report: {output_path}")

    if summary["errors"]:
        return 1
    if args.strict and summary["warnings"]:
        return 1
    if args.certify_candidate:
        assert candidate_dir is not None
        certification = certify_candidate_audit(candidate_dir, output_path)
        print(f"Audit certification: {candidate_dir / 'AUDITED'}")
        print(f"Certified report SHA-256: {certification['report_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
