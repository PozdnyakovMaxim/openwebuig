from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from docx import Document

from document_search.chunker import chunk_document
from document_search import corpus_audit as corpus_audit_module
from document_search.corpus_audit import audit_corpus
from document_search.extractor import extract_docx, write_extraction


class CorpusAuditTest(unittest.TestCase):
    def test_clean_artifacts_pass(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            paths = self._build_corpus(Path(temporary_directory))
            report = audit_corpus(*paths, skip_database=True)

        self.assertEqual(report["status"], "ok", report["issues"])
        self.assertEqual(report["summary"]["source_documents"], 1)
        self.assertEqual(report["summary"]["errors"], 0)
        self.assertEqual(
            report["documents"][0]["covered_blocks"],
            report["documents"][0]["indexable_blocks"],
        )
        self.assertEqual(report["documents"][0]["chunk_text_coverage"], 1.0)
        self.assertEqual(report["documents"][0]["chunk_snapshot_changed"], 0)
        self.assertEqual(report["summary"]["unsearchable_non_indexed_blocks"], 0)

    def test_missing_block_is_reported(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            paths = self._build_corpus(Path(temporary_directory))
            chunks_path = next(paths[2].glob("*.chunks.json"))
            data = json.loads(chunks_path.read_text(encoding="utf-8"))
            data["chunks"] = [
                chunk
                for chunk in data["chunks"]
                if "item-1.1.2" not in (chunk.get("block_ids") or [])
            ]
            data["chunk_count"] = len(data["chunks"])
            chunks_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            manifest_path = paths[2] / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest[0]["chunks"] = len(data["chunks"])
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

            report = audit_corpus(*paths, skip_database=True)

        codes = {issue["code"] for issue in report["issues"]}
        self.assertEqual(report["status"], "error")
        self.assertIn("chunk_blocks_missing", codes)
        self.assertIn("chunk_text_coverage_low", codes)

    def test_stale_extraction_is_reported(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            paths = self._build_corpus(Path(temporary_directory))
            extraction_path = next(paths[1].glob("*.json"))
            if extraction_path.name == "manifest.json":
                extraction_path = next(
                    path for path in paths[1].glob("*.json") if path.name != "manifest.json"
                )
            data = json.loads(extraction_path.read_text(encoding="utf-8"))
            data["blocks"] = [
                block for block in data["blocks"] if block.get("block_id") != "item-1.1.2"
            ]
            data["block_count"] = len(data["blocks"])
            extraction_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

            report = audit_corpus(*paths, skip_database=True)

        codes = {issue["code"] for issue in report["issues"]}
        self.assertEqual(report["status"], "error")
        self.assertIn("extraction_blocks_missing", codes)

    def test_empty_source_directory_is_reported(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            docs_dir = root / "docs"
            extracted_dir = root / "extracted"
            chunks_dir = root / "chunks"
            for directory in (docs_dir, extracted_dir, chunks_dir):
                directory.mkdir()
            for directory in (extracted_dir, chunks_dir):
                (directory / "manifest.json").write_text("[]", encoding="utf-8")

            report = audit_corpus(
                docs_dir,
                extracted_dir,
                chunks_dir,
                skip_database=True,
            )

        codes = {issue["code"] for issue in report["issues"]}
        self.assertEqual(report["status"], "error")
        self.assertIn("source_documents_missing", codes)

    def test_missing_metadata_chunk_is_reported(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            paths = self._build_corpus(Path(temporary_directory))
            chunks_path = next(paths[2].glob("*.chunks.json"))
            data = json.loads(chunks_path.read_text(encoding="utf-8"))
            data["chunks"] = [
                chunk for chunk in data["chunks"] if chunk.get("chunk_type") != "metadata"
            ]
            data["chunk_count"] = len(data["chunks"])
            chunks_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            manifest_path = paths[2] / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest[0]["chunks"] = len(data["chunks"])
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

            report = audit_corpus(*paths, skip_database=True)

        codes = {issue["code"] for issue in report["issues"]}
        self.assertEqual(report["status"], "error")
        self.assertIn("metadata_chunk_count_invalid", codes)

    def test_changed_searchable_text_is_reported(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            paths = self._build_corpus(Path(temporary_directory))
            chunks_path = next(paths[2].glob("*.chunks.json"))
            data = json.loads(chunks_path.read_text(encoding="utf-8"))
            data["chunks"][-1]["searchable_text"] += " устаревшее значение"
            chunks_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

            report = audit_corpus(*paths, skip_database=True)

        codes = {issue["code"] for issue in report["issues"]}
        self.assertEqual(report["status"], "error")
        self.assertIn("chunk_snapshot_changed", codes)

    def test_heading_missing_from_chunks_is_reported(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            paths = self._build_corpus(Path(temporary_directory))
            extraction_path = next(
                path for path in paths[1].glob("*.json") if path.name != "manifest.json"
            )
            extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
            heading_id = next(
                block["block_id"]
                for block in extraction["blocks"]
                if block.get("kind") == "heading" and block.get("text") == "Назначение"
            )
            chunks_path = next(paths[2].glob("*.chunks.json"))
            data = json.loads(chunks_path.read_text(encoding="utf-8"))
            data["chunks"] = [
                chunk
                for chunk in data["chunks"]
                if heading_id not in (chunk.get("block_ids") or [])
            ]
            data["chunk_count"] = len(data["chunks"])
            chunks_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            manifest_path = paths[2] / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest[0]["chunks"] = len(data["chunks"])
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

            report = audit_corpus(*paths, skip_database=True)

        codes = {issue["code"] for issue in report["issues"]}
        self.assertEqual(report["status"], "error")
        self.assertIn("chunk_blocks_missing", codes)

    def test_custom_chunk_size_is_replayed_from_manifest(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            paths = self._build_corpus(Path(temporary_directory), max_chars=24)
            report = audit_corpus(*paths, skip_database=True)

        self.assertEqual(report["status"], "ok", report["issues"])
        self.assertEqual(report["documents"][0]["chunk_max_chars"], 24)
        self.assertEqual(report["documents"][0]["chunk_snapshot_changed"], 0)

    def test_database_chunk_without_embedding_is_reported(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            paths = self._build_corpus(Path(temporary_directory))
            extraction_path = next(
                path for path in paths[1].glob("*.json") if path.name != "manifest.json"
            )
            extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
            chunk_path = next(paths[2].glob("*.chunks.json"))
            chunk = json.loads(chunk_path.read_text(encoding="utf-8"))["chunks"][0]

            document_row = {
                "doc_id": extraction["metadata"]["doc_id"],
                "source_name": extraction["metadata"]["source_name"],
                "metadata": extraction["metadata"],
            }
            chunk_row = {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "source_name": chunk["source_name"],
                "raw_text": chunk["raw_text"],
                "searchable_text": chunk["searchable_text"],
                "block_ids": chunk.get("block_ids") or [],
                "char_count": chunk["char_count"],
                "embedding_model": "bge-m3",
                "metadata": chunk,
                "embedding_missing": True,
                "embedding_dim": None,
            }
            connection = _FakeConnection(document_row, chunk_row)
            issues: list[dict[str, object]] = []
            with (
                patch.object(corpus_audit_module, "database_url", return_value="postgresql://db"),
                patch.object(corpus_audit_module, "connect", return_value=connection),
            ):
                result = corpus_audit_module._audit_database(
                    None,
                    {document_row["doc_id"]: document_row},
                    {chunk_row["chunk_id"]: chunk},
                    issues,
                )

        self.assertTrue(result["checked"])
        self.assertEqual(result["missing_embeddings"], 1)
        self.assertIn("database_embeddings_missing", {issue["code"] for issue in issues})

    def _build_corpus(
        self,
        root: Path,
        *,
        max_chars: int = 1800,
    ) -> tuple[Path, Path, Path]:
        docs_dir = root / "docs"
        extracted_dir = root / "artifacts" / "extracted"
        chunks_dir = root / "artifacts" / "chunks"
        docs_dir.mkdir(parents=True)
        extracted_dir.mkdir(parents=True)
        chunks_dir.mkdir(parents=True)

        source_path = docs_dir / "Политика.docx"
        document = Document()
        document.add_paragraph("ИНДЕКС НД: TEST-1.0")
        document.add_paragraph("ПОЛИТИКА")
        document.add_paragraph("1 Общие положения")
        document.add_paragraph("1.1 Назначение")
        document.add_paragraph("1.1.1 Первый обязательный пункт документа.")
        document.add_paragraph("1.1.2 Второй обязательный пункт документа.")
        document.save(source_path)

        extracted = extract_docx(source_path)
        extracted_path = extracted_dir / f"{extracted.metadata.doc_id}.json"
        write_extraction(extracted, extracted_path)
        extraction_manifest = [
            {
                "source_name": extracted.metadata.source_name,
                "source_path": extracted.metadata.source_path,
                "doc_id": extracted.metadata.doc_id,
                "output_path": str(extracted_path),
            }
        ]
        (extracted_dir / "manifest.json").write_text(
            json.dumps(extraction_manifest, ensure_ascii=False),
            encoding="utf-8",
        )

        chunked = chunk_document(extracted.to_dict(), max_chars=max_chars)
        chunks_path = chunks_dir / f"{extracted.metadata.doc_id}.chunks.json"
        chunks_path.write_text(json.dumps(chunked, ensure_ascii=False), encoding="utf-8")
        chunk_manifest = [
            {
                "doc_id": extracted.metadata.doc_id,
                "source_name": extracted.metadata.source_name,
                "extracted_path": str(extracted_path),
                "chunked_path": str(chunks_path),
                "chunks": chunked["chunk_count"],
                "max_chars": max_chars,
            }
        ]
        (chunks_dir / "manifest.json").write_text(
            json.dumps(chunk_manifest, ensure_ascii=False),
            encoding="utf-8",
        )
        return docs_dir, extracted_dir, chunks_dir


class _FakeRows:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def fetchall(self) -> list[dict[str, object]]:
        return self.rows


class _FakeConnection:
    def __init__(self, document: dict[str, object], chunk: dict[str, object]) -> None:
        self.document = document
        self.chunk = chunk

    def __enter__(self) -> _FakeConnection:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, query: str) -> _FakeRows:
        normalized = " ".join(query.split())
        if "FROM doc_documents" in normalized:
            return _FakeRows([self.document])
        if "FROM doc_chunks" in normalized:
            return _FakeRows([self.chunk])
        if "FROM pg_indexes" in normalized:
            return _FakeRows([{"indexname": "doc_chunks_embedding_hnsw"}])
        raise AssertionError(f"Unexpected query: {normalized}")


if __name__ == "__main__":
    unittest.main()
