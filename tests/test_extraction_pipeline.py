from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from docx import Document

from document_search.chunker import chunk_document
from document_search.extractor import extract_docx


class ExtractionPipelineTest(unittest.TestCase):
    def test_first_paragraph_after_toc_is_retained(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "toc.docx"
            document = Document()
            document.add_paragraph("СОДЕРЖАНИЕ")
            document.add_paragraph("1 Общие положения ........ 3")
            document.add_paragraph("После содержания этот абзац должен сохраниться.")
            document.save(source_path)

            extracted = extract_docx(source_path).to_dict()

        texts = [block["text"] for block in extracted["blocks"]]
        self.assertIn("После содержания этот абзац должен сохраниться.", texts)

    def test_repeated_numbered_items_receive_unique_block_ids(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "duplicates.docx"
            document = Document()
            document.add_paragraph("1.1.1 Первый пункт.")
            document.add_paragraph("1.1.1 Второй пункт.")
            document.save(source_path)

            extracted = extract_docx(source_path).to_dict()

        block_ids = [block["block_id"] for block in extracted["blocks"]]
        self.assertEqual(len(block_ids), len(set(block_ids)))
        self.assertTrue(any(block_id.endswith("--2") for block_id in block_ids))

    def test_every_substantive_block_is_present_in_chunks(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "coverage.docx"
            document = Document()
            document.add_paragraph("ИНДЕКС НД: TEST-1.0")
            document.add_paragraph("ПОЛИТИКА")
            document.add_paragraph("1 Общие положения")
            document.add_paragraph("1.1 Назначение")
            document.add_paragraph("1.1.1 Первый обязательный пункт.")
            document.add_paragraph("Обычный поясняющий абзац.")
            document.save(source_path)

            extracted = extract_docx(source_path).to_dict()
            chunked = chunk_document(extracted)

        expected = {
            block["block_id"]
            for block in extracted["blocks"]
            if str(block.get("text") or "").strip()
        }
        covered = {
            block_id
            for chunk in chunked["chunks"]
            for block_id in chunk.get("block_ids") or []
            if block_id != "metadata"
        }
        self.assertEqual(expected, covered)


if __name__ == "__main__":
    unittest.main()
