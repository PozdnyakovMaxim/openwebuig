from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from docx import Document


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from rebuild_corpus_candidate import build_candidate
from step3_index_chunks import (
    load_and_validate_corpus,
    load_candidate_marker,
    resolve_chunk_files,
    validate_embeddings,
)


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
            self.assertTrue((output_dir / "READY").is_file())
            self.assertEqual(load_candidate_marker(output_dir / "chunks"), summary)
            files = resolve_chunk_files(output_dir / "chunks")
            records = load_and_validate_corpus(files, expected_documents=2)
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


if __name__ == "__main__":
    unittest.main()
