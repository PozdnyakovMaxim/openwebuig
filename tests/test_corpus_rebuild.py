from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from docx import Document


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from rebuild_corpus_candidate import build_candidate
from step3_index_chunks import (
    CORPUS_PROMOTION_ADVISORY_LOCK_KEY,
    acquire_corpus_promotion_lock,
    load_and_validate_corpus,
    load_candidate_marker,
    resolve_embedding_index_id,
    resolve_chunk_files,
    validate_embeddings,
)
from corpus_candidate_integrity import (
    certify_candidate_audit,
    sha256_file,
    verify_candidate_audit,
    verify_candidate_integrity,
)
from document_search.corpus_audit import audit_corpus


class CorpusCandidateTest(unittest.TestCase):
    def _write_document(self, path: Path, paragraphs: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        document = Document()
        for paragraph in paragraphs:
            document.add_paragraph(paragraph)
        document.save(path)

    def test_candidate_is_complete_and_has_ready_marker(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            docs_dir = root / "docs"
            output_dir = root / "candidate"
            self._write_document(
                docs_dir / "first.docx",
                ["1 Общие положения", "1.1 Первый обязательный пункт."],
            )
            self._write_document(
                docs_dir / "nested" / "second.docx",
                ["2 Требования", "2.1 Второй обязательный пункт."],
            )

            summary = build_candidate(
                docs_dir,
                output_dir,
                expected_documents=2,
            )

            self.assertEqual(summary["documents"], 2)
            self.assertEqual(summary["schema_version"], 1)
            self.assertTrue(summary["tool_version"])
            self.assertTrue((output_dir / "READY").is_file())
            self.assertEqual(load_candidate_marker(output_dir / "chunks"), summary)
            verify_candidate_integrity(output_dir, summary)
            audit_report = audit_corpus(
                docs_dir,
                output_dir / "extracted",
                output_dir / "chunks",
                skip_database=True,
            )
            audit_report["candidate"] = {
                "candidate_dir": str(output_dir.resolve()),
                "ready_sha256": sha256_file(output_dir / "READY"),
            }
            audit_path = output_dir / "audit-before.json"
            audit_path.write_text(
                json.dumps(audit_report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            certify_candidate_audit(output_dir, audit_path)
            verify_candidate_audit(output_dir, summary)
            self.assertTrue((output_dir / "AUDITED").is_file())
            hashed_kinds = {
                item["kind"] for item in summary["integrity"]["files"]
            }
            self.assertEqual(
                hashed_kinds,
                {
                    "source_docx",
                    "extracted_json",
                    "chunk_json",
                    "extraction_manifest",
                    "chunk_manifest",
                },
            )
            files = resolve_chunk_files(
                output_dir / "chunks",
                candidate_marker=summary,
            )
            expected_hashes = {
                (output_dir / item["path"]).resolve(): item["sha256"]
                for item in summary["integrity"]["files"]
                if item["kind"] == "chunk_json"
            }
            records = load_and_validate_corpus(
                files,
                expected_documents=2,
                expected_file_hashes=expected_hashes,
            )
            self.assertEqual(len(records), 2)

            for record in records:
                extracted_path = output_dir / "extracted" / f"{record['metadata']['doc_id']}.json"
                extracted = json.loads(extracted_path.read_text(encoding="utf-8"))
                expected = {
                    block["block_id"]
                    for block in extracted["blocks"]
                    if str(block.get("text") or "").strip()
                }
                covered = {
                    block_id
                    for chunk in record["chunks"]
                    for block_id in chunk.get("block_ids") or []
                    if block_id != "metadata"
                }
                self.assertEqual(expected, covered)

    def test_failed_validation_does_not_publish_candidate(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            docs_dir = root / "docs"
            output_dir = root / "candidate"
            self._write_document(docs_dir / "only.docx", ["Содержательный текст документа."])

            with self.assertRaisesRegex(ValueError, "Document count mismatch"):
                build_candidate(docs_dir, output_dir, expected_documents=2)

            self.assertFalse(output_dir.exists())
            self.assertEqual(list(root.glob(".candidate.staging-*")), [])

    def test_existing_output_file_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            docs_dir = root / "docs"
            output_path = root / "candidate"
            self._write_document(docs_dir / "only.docx", ["Содержательный текст документа."])
            output_path.write_text("do not overwrite", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Output path is not a directory"):
                build_candidate(docs_dir, output_path, expected_documents=1)

            self.assertEqual(output_path.read_text(encoding="utf-8"), "do not overwrite")
            self.assertEqual(list(root.glob(".candidate.staging-*")), [])

    def test_duplicate_chunk_ids_are_rejected(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths: list[Path] = []
            for index in range(2):
                path = root / f"doc-{index}.chunks.json"
                payload = {
                    "metadata": {
                        "doc_id": f"doc-{index}",
                        "source_name": f"doc-{index}.docx",
                    },
                    "chunk_count": 1,
                    "chunks": [
                        {
                            "chunk_id": "duplicate-chunk",
                            "doc_id": f"doc-{index}",
                            "raw_text": "Текст",
                            "searchable_text": "Текст",
                            "char_count": 5,
                        }
                    ],
                }
                path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                paths.append(path)

            with self.assertRaisesRegex(ValueError, "Duplicate chunk ID"):
                load_and_validate_corpus(paths, expected_documents=2)

    def test_invalid_embedding_dimension_is_rejected(self) -> None:
        records = [
            {
                "embedded_chunks": [
                    (
                        {"chunk_id": "chunk-1"},
                        [0.1, 0.2],
                    )
                ]
            }
        ]

        with self.assertRaisesRegex(ValueError, "expected embedding dimension 3"):
            validate_embeddings(records, embedding_dim=3)

    def test_tampered_candidate_artifact_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            docs_dir = root / "docs"
            output_dir = root / "candidate"
            self._write_document(docs_dir / "only.docx", ["Содержательный текст документа."])
            summary = build_candidate(docs_dir, output_dir, expected_documents=1)
            chunk_path = next((output_dir / "chunks").glob("*.chunks.json"))
            chunk_path.write_text(
                chunk_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "integrity check failed.*chunk_json"):
                verify_candidate_integrity(output_dir, summary)

    def test_loaded_candidate_bytes_must_match_ready_hash(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            docs_dir = root / "docs"
            output_dir = root / "candidate"
            self._write_document(docs_dir / "only.docx", ["Содержательный текст документа."])
            summary = build_candidate(docs_dir, output_dir, expected_documents=1)
            files = resolve_chunk_files(
                output_dir / "chunks",
                candidate_marker=summary,
            )
            expected_hashes = {
                (output_dir / item["path"]).resolve(): item["sha256"]
                for item in summary["integrity"]["files"]
                if item["kind"] == "chunk_json"
            }
            files[0].write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Loaded chunk file hash mismatch"):
                load_and_validate_corpus(
                    files,
                    expected_documents=1,
                    expected_file_hashes=expected_hashes,
                )

    def test_changed_source_document_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            docs_dir = root / "docs"
            source_path = docs_dir / "only.docx"
            output_dir = root / "candidate"
            self._write_document(source_path, ["Исходная редакция документа."])
            summary = build_candidate(docs_dir, output_dir, expected_documents=1)
            self._write_document(source_path, ["Изменённая редакция документа."])

            with self.assertRaisesRegex(ValueError, "integrity check failed.*source_docx"):
                verify_candidate_integrity(output_dir, summary)

    def test_new_source_document_after_candidate_build_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            docs_dir = root / "docs"
            output_dir = root / "candidate"
            self._write_document(docs_dir / "first.docx", ["Первый документ."])
            summary = build_candidate(docs_dir, output_dir, expected_documents=1)
            self._write_document(docs_dir / "second.docx", ["Новый документ."])

            with self.assertRaisesRegex(ValueError, "source inventory changed"):
                verify_candidate_integrity(output_dir, summary)

    def test_source_change_between_extraction_and_manifest_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            docs_dir = root / "docs"
            source_path = docs_dir / "only.docx"
            output_dir = root / "candidate"
            self._write_document(source_path, ["Исходная редакция."])

            from rebuild_corpus_candidate import build_integrity_manifest as original

            def mutate_source(**kwargs):
                self._write_document(source_path, ["Изменённая редакция."])
                return original(**kwargs)

            with (
                patch(
                    "rebuild_corpus_candidate.build_integrity_manifest",
                    side_effect=mutate_source,
                ),
                self.assertRaisesRegex(ValueError, "source hash mismatch"),
            ):
                build_candidate(docs_dir, output_dir, expected_documents=1)

            self.assertFalse(output_dir.exists())
            self.assertEqual(list(root.glob(".candidate.staging-*")), [])

    def test_candidate_manifest_cannot_redirect_to_external_chunks(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            docs_dir = root / "docs"
            output_dir = root / "candidate"
            self._write_document(docs_dir / "only.docx", ["Содержательный текст."])
            summary = build_candidate(docs_dir, output_dir, expected_documents=1)
            external = root / "external.chunks.json"
            external.write_text("{}", encoding="utf-8")
            manifest_path = output_dir / "chunks" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest[0]["chunked_path"] = str(external.resolve())
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "must be relative"):
                resolve_chunk_files(
                    output_dir / "chunks",
                    candidate_marker=summary,
                )

    def test_strict_audit_report_must_match_the_exact_ready_digest(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            docs_dir = root / "docs"
            output_dir = root / "candidate"
            self._write_document(docs_dir / "only.docx", ["Содержательный текст."])
            build_candidate(docs_dir, output_dir, expected_documents=1)
            report = audit_corpus(
                docs_dir,
                output_dir / "extracted",
                output_dir / "chunks",
                skip_database=True,
            )
            report["candidate"] = {
                "candidate_dir": str(output_dir.resolve()),
                "ready_sha256": "0" * 64,
            }
            report_path = output_dir / "audit-before.json"
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "READY digest does not match"):
                certify_candidate_audit(output_dir, report_path)

    def test_atomic_promotion_issues_transaction_advisory_lock(self) -> None:
        class Connection:
            def __init__(self) -> None:
                self.calls: list[tuple[str, tuple[int, ...]]] = []

            def execute(self, query: str, parameters: tuple[int, ...]) -> None:
                self.calls.append((query, parameters))

        connection = Connection()
        acquire_corpus_promotion_lock(connection)

        self.assertEqual(
            connection.calls,
            [
                (
                    "SELECT pg_advisory_xact_lock(%s::bigint)",
                    (CORPUS_PROMOTION_ADVISORY_LOCK_KEY,),
                )
            ],
        )

    def test_stable_embedder_index_id_is_preferred_over_local_model_path(self) -> None:
        class Embedder:
            model = "/opt/models/bge-m3"
            index_id = "BAAI/bge-m3"

        self.assertEqual(resolve_embedding_index_id(Embedder()), "BAAI/bge-m3")

    def test_embedder_model_is_used_as_compatibility_fallback(self) -> None:
        class Embedder:
            model = "provider/model"

        self.assertEqual(resolve_embedding_index_id(Embedder()), "provider/model")


if __name__ == "__main__":
    unittest.main()
