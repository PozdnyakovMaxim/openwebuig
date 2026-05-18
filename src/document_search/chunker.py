from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MAX_CHARS = 1800
DEFAULT_MIN_PACK_CHARS = 320


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source_name: str
    index_code: str | None
    document_title: str | None
    version: str | None
    chunk_type: str
    citation_label: str
    raw_text: str
    searchable_text: str
    block_ids: list[str]
    section_path: list[str] = field(default_factory=list)
    section_title: str | None = None
    subsection_title: str | None = None
    item_number: str | None = None
    appendix_number: str | None = None
    appendix_title: str | None = None
    char_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChunkUnit:
    unit_type: str
    blocks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def block_ids(self) -> list[str]:
        return [str(block["block_id"]) for block in self.blocks]

    @property
    def raw_text(self) -> str:
        return "\n".join(_format_block_text(block) for block in self.blocks if block.get("text")).strip()

    @property
    def first_block(self) -> dict[str, Any]:
        return self.blocks[0]

    @property
    def item_number(self) -> str | None:
        for block in self.blocks:
            if block.get("item_number"):
                return block["item_number"]
        return None

    @property
    def section_path(self) -> list[str]:
        return list(self.first_block.get("section_path") or [])

    @property
    def appendix_number(self) -> str | None:
        return self.first_block.get("appendix_number")

    @property
    def appendix_title(self) -> str | None:
        return self.first_block.get("appendix_title")

    def can_absorb(self, block: dict[str, Any], max_chars: int) -> bool:
        if not self.blocks:
            return True
        if self.appendix_number != block.get("appendix_number"):
            return False
        if self.section_path != list(block.get("section_path") or []):
            return False
        candidate = self.raw_text + "\n" + _format_block_text(block)
        return len(candidate) <= max_chars


def _stable_hash(text: str, length: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def _normalize_space(text: str) -> str:
    cleaned = text.replace("\xa0", " ").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _one_line(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _format_block_text(block: dict[str, Any]) -> str:
    text = _normalize_space(str(block.get("text") or ""))
    kind = block.get("kind")
    if kind in {"numbered_paragraph", "appendix_numbered_item"} and block.get("item_number"):
        return f"{block['item_number']} {text}"
    if kind in {"letter_bullet", "appendix_bullet"} and block.get("item_marker"):
        return f"{block['item_marker']}) {text}"
    return text


def _metadata_text(metadata: dict[str, Any]) -> str:
    lines = [
        ("Документ", metadata.get("display_title")),
        ("Индекс", metadata.get("index_code")),
        ("Тип", metadata.get("document_kind")),
        ("Организация", metadata.get("organization")),
        ("Версия", metadata.get("version")),
        ("Дата утверждения", metadata.get("approval_date")),
        ("Номер приказа", metadata.get("approval_order_number")),
        ("Дата ввода в действие", metadata.get("effective_date")),
        ("Листов", metadata.get("declared_pages")),
        ("Файл", metadata.get("source_name")),
    ]
    return "\n".join(f"{label}: {value}" for label, value in lines if value not in (None, "", []))


def _section_label(unit: ChunkUnit) -> str:
    parts: list[str] = []
    block = unit.first_block
    if block.get("section_number"):
        value = _one_line(block["section_number"])
        if block.get("section_title"):
            value += f" {_one_line(block['section_title'])}"
        parts.append(value)
    if block.get("subsection_number"):
        value = _one_line(block["subsection_number"])
        if block.get("subsection_title"):
            value += f" {_one_line(block['subsection_title'])}"
        parts.append(value)
    return " / ".join(parts)


def _citation_label(metadata: dict[str, Any], unit: ChunkUnit, part: int | None = None) -> str:
    title = _one_line(metadata.get("display_title") or metadata.get("source_name") or metadata.get("doc_id"))
    version = metadata.get("version")
    base = f"{title}"
    if version:
        base += f" (версия {_one_line(version)})"

    if unit.appendix_number:
        base += f", приложение № {_one_line(unit.appendix_number)}"
        if unit.appendix_title:
            base += f" {_one_line(unit.appendix_title)}"
        if unit.item_number:
            base += f", пункт {_one_line(unit.item_number)}"
    else:
        section_label = _section_label(unit)
        if section_label:
            base += f", раздел {section_label}"
        if unit.item_number:
            base += f", пункт {_one_line(unit.item_number)}"

    if part is not None:
        base += f", часть {part}"
    return base


def _context_text(metadata: dict[str, Any], unit: ChunkUnit) -> str:
    lines = []
    if metadata.get("display_title"):
        lines.append(f"Документ: {_one_line(metadata['display_title'])}")
    if metadata.get("index_code"):
        lines.append(f"Индекс: {_one_line(metadata['index_code'])}")
    if metadata.get("version"):
        lines.append(f"Версия: {_one_line(metadata['version'])}")
    if unit.appendix_number:
        appendix = f"Приложение № {_one_line(unit.appendix_number)}"
        if unit.appendix_title:
            appendix += f": {_one_line(unit.appendix_title)}"
        lines.append(appendix)
    else:
        section_label = _section_label(unit)
        if section_label:
            lines.append(f"Раздел: {section_label}")
        if unit.item_number:
            lines.append(f"Пункт: {unit.item_number}")
    return "\n".join(lines)


def _split_text(text: str, max_chars: int) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    paragraphs = [part.strip() for part in text.split("\n") if part.strip()]
    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        extra = len(paragraph) + (1 if current else 0)
        if current and current_len + extra > max_chars:
            parts.append("\n".join(current))
            current = []
            current_len = 0
        if len(paragraph) > max_chars:
            sentences = re.split(r"(?<=[.!?;:])\s+", paragraph)
            for sentence in sentences:
                if not sentence:
                    continue
                if len(sentence) > max_chars:
                    if current:
                        parts.append("\n".join(current))
                        current = []
                        current_len = 0
                    parts.extend(_hard_split(sentence, max_chars))
                    continue
                extra_sentence = len(sentence) + (1 if current else 0)
                if current and current_len + extra_sentence > max_chars:
                    parts.append("\n".join(current))
                    current = []
                    current_len = 0
                current.append(sentence)
                current_len += extra_sentence
            continue
        current.append(paragraph)
        current_len += extra
    if current:
        parts.append("\n".join(current))
    return parts or [text]


def _hard_split(text: str, max_chars: int) -> list[str]:
    return [text[index : index + max_chars].strip() for index in range(0, len(text), max_chars) if text[index : index + max_chars].strip()]


def _make_chunk(
    metadata: dict[str, Any],
    unit: ChunkUnit,
    raw_text: str,
    ordinal: int,
    *,
    part: int | None = None,
) -> Chunk:
    context = _context_text(metadata, unit)
    searchable_text = _normalize_space("\n\n".join(part for part in (context, raw_text) if part))
    chunk_source = "|".join(
        [
            str(metadata.get("doc_id") or metadata.get("source_name")),
            ",".join(unit.block_ids),
            str(part or 0),
            raw_text,
        ]
    )
    chunk_id = f"{metadata.get('doc_id') or 'document'}::chunk-{ordinal:04d}-{_stable_hash(chunk_source)}"
    first = unit.first_block
    return Chunk(
        chunk_id=chunk_id,
        doc_id=str(metadata.get("doc_id") or ""),
        source_name=str(metadata.get("source_name") or ""),
        index_code=metadata.get("index_code"),
        document_title=metadata.get("display_title"),
        version=metadata.get("version"),
        chunk_type=unit.unit_type,
        citation_label=_citation_label(metadata, unit, part=part),
        raw_text=raw_text,
        searchable_text=searchable_text,
        block_ids=unit.block_ids,
        section_path=unit.section_path,
        section_title=_one_line(first.get("section_title")) or None,
        subsection_title=_one_line(first.get("subsection_title")) or None,
        item_number=unit.item_number,
        appendix_number=unit.appendix_number,
        appendix_title=_one_line(unit.appendix_title) or None,
        char_count=len(raw_text),
    )


def _flush_unit(
    chunks: list[Chunk],
    metadata: dict[str, Any],
    unit: ChunkUnit | None,
    max_chars: int,
) -> None:
    if not unit or not unit.blocks or not unit.raw_text:
        return
    parts = _split_text(unit.raw_text, max_chars)
    for part_index, part_text in enumerate(parts, start=1):
        chunks.append(
            _make_chunk(
                metadata,
                unit,
                part_text,
                len(chunks) + 1,
                part=part_index if len(parts) > 1 else None,
            )
        )


def _same_parent_item(unit: ChunkUnit | None, block: dict[str, Any]) -> bool:
    if unit is None or not unit.blocks:
        return False
    if unit.unit_type not in {"numbered_item", "appendix_item"}:
        return False
    if unit.appendix_number != block.get("appendix_number"):
        return False
    if unit.section_path != list(block.get("section_path") or []):
        return False
    return True


def chunk_document(
    extracted_document: dict[str, Any],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> dict[str, Any]:
    metadata = extracted_document["metadata"]
    blocks: list[dict[str, Any]] = extracted_document.get("blocks") or []
    chunks: list[Chunk] = []

    meta_text = _metadata_text(metadata)
    if meta_text:
        meta_unit = ChunkUnit(
            unit_type="metadata",
            blocks=[
                {
                    "block_id": "metadata",
                    "kind": "metadata",
                    "text": meta_text,
                    "source_kind": "metadata",
                    "section_path": [],
                }
            ],
        )
        _flush_unit(chunks, metadata, meta_unit, max_chars)

    current: ChunkUnit | None = None

    for block in blocks:
        kind = block.get("kind")

        if kind in {"front_matter", "heading", "appendix_heading", "appendix_title"}:
            _flush_unit(chunks, metadata, current, max_chars)
            current = None
            continue

        if kind == "numbered_paragraph":
            _flush_unit(chunks, metadata, current, max_chars)
            current = ChunkUnit(unit_type="numbered_item", blocks=[block])
            continue

        if kind == "appendix_numbered_item":
            _flush_unit(chunks, metadata, current, max_chars)
            current = ChunkUnit(unit_type="appendix_item", blocks=[block])
            continue

        if kind in {"letter_bullet", "appendix_bullet"}:
            if _same_parent_item(current, block) and current.can_absorb(block, max_chars):
                current.blocks.append(block)
            else:
                _flush_unit(chunks, metadata, current, max_chars)
                unit_type = "appendix_list" if block.get("appendix_number") else "list"
                current = ChunkUnit(unit_type=unit_type, blocks=[block])
            continue

        if kind in {"paragraph", "appendix_paragraph"}:
            unit_type = "appendix_text" if block.get("appendix_number") else "section_text"
            if current and current.can_absorb(block, max_chars):
                current.blocks.append(block)
            else:
                _flush_unit(chunks, metadata, current, max_chars)
                current = ChunkUnit(unit_type=unit_type, blocks=[block])
            continue

        if block.get("text"):
            if current and current.can_absorb(block, max_chars):
                current.blocks.append(block)
            else:
                _flush_unit(chunks, metadata, current, max_chars)
                current = ChunkUnit(unit_type="other", blocks=[block])

    _flush_unit(chunks, metadata, current, max_chars)

    return {
        "metadata": metadata,
        "chunk_count": len(chunks),
        "chunks": [chunk.to_dict() for chunk in chunks],
    }


def chunk_extracted_file(
    source_path: str | Path,
    destination: str | Path,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> Path:
    source = Path(source_path)
    extracted = json.loads(source.read_text(encoding="utf-8"))
    chunked = chunk_document(extracted, max_chars=max_chars)
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(chunked, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def chunk_many(
    source_paths: Iterable[str | Path],
    output_dir: str | Path,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[dict[str, Any]]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for source_path in source_paths:
        source = Path(source_path)
        extracted = json.loads(source.read_text(encoding="utf-8"))
        doc_id = extracted["metadata"].get("doc_id") or source.stem
        destination = output / f"{doc_id}.chunks.json"
        chunked = chunk_document(extracted, max_chars=max_chars)
        destination.write_text(json.dumps(chunked, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest.append(
            {
                "doc_id": doc_id,
                "source_name": extracted["metadata"].get("source_name"),
                "extracted_path": str(source),
                "chunked_path": str(destination),
                "chunks": chunked["chunk_count"],
                "max_chars": max_chars,
            }
        )
    return manifest
