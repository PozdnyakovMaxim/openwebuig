from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import re
from typing import Iterable

from docx import Document
from docx.document import Document as DocumentObject
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph


DATE_RE = re.compile(r"\b\d{2}\.\d{2}\.\d{4}\b")
ORG_RE = re.compile(r"(?:ООО|АО|ПАО)\s+«[^»]+»", re.IGNORECASE)
INDEX_RE = re.compile(
    r"(?:ИНДЕКС|Индекс)\s+(?:НД|ЛНА)\s*[:：]?\s*(?P<value>[A-ZА-ЯЁ0-9./-]+)",
    re.IGNORECASE,
)
VERSION_RE = re.compile(r"ВЕРСИЯ\s*(?P<value>[\d.]+)", re.IGNORECASE)
EFFECTIVE_DATE_RE = re.compile(
    r"Введено\s+в\s+действие\s+с\s*[:：]?\s*(?P<date>\d{2}\.\d{2}\.\d{4})",
    re.IGNORECASE,
)
DECLARED_PAGES_RE = re.compile(r"Листов\s*[:：]?\s*(?P<count>\d+)", re.IGNORECASE)
APPROVAL_ORDER_RE = re.compile(
    r"от\s*(?P<date>\d{2}\.\d{2}\.\d{4})\s*№\s*(?P<number>.+)$",
    re.IGNORECASE,
)

TOC_TITLE_RE = re.compile(r"^СОДЕРЖАНИЕ$", re.IGNORECASE)
TOC_ENTRY_RE = re.compile(r"(?:\t|\s{2,}|\.+)\d+$|[А-Яа-яA-Za-zЁё)]\d+$")
APPENDIX_RE = re.compile(
    r"^Приложение\s*№?\s*(?P<number>\d+)(?:\s+(?P<title>.+))?$",
    re.IGNORECASE,
)

NUMERIC_PREFIX_RE = re.compile(r"^(?P<number>\d+(?:\.\d+)*)(?:[.)])?\s+(?P<text>.+)$")
LETTER_BULLET_RE = re.compile(r"^(?P<marker>[А-Яа-яЁёA-Za-z])\)\s+(?P<text>.+)$")
HEADING_STYLE_RE = re.compile(r"^(?:Heading|Заголовок)\s*(?P<level>\d+)?$", re.IGNORECASE)
TOC_STYLE_RE = re.compile(r"^(?:TOC|Оглавление|Содержание)", re.IGNORECASE)
DOCUMENT_KIND_RE = re.compile(
    r"^(?:ПОЛИТИКА|СТАНДАРТ|РЕГЛАМЕНТ|ИНСТРУКЦИЯ|ПОЛОЖЕНИЕ|ПОРЯДОК|ПРОЦЕДУРА)$",
    re.IGNORECASE,
)


@dataclass
class RawBlock:
    source_kind: str
    text: str
    style_name: str | None = None
    is_bold: bool = False


@dataclass
class DocumentMetadata:
    source_path: str
    source_name: str
    doc_id: str
    index_code: str | None = None
    display_title: str | None = None
    document_kind: str | None = None
    organization: str | None = None
    version: str | None = None
    approval_date: str | None = None
    approval_order_number: str | None = None
    effective_date: str | None = None
    declared_pages: int | None = None
    title_lines: list[str] = field(default_factory=list)


@dataclass
class ContentBlock:
    block_id: str
    kind: str
    text: str
    source_kind: str
    section_number: str | None = None
    section_title: str | None = None
    subsection_number: str | None = None
    subsection_title: str | None = None
    item_number: str | None = None
    item_marker: str | None = None
    appendix_number: str | None = None
    appendix_title: str | None = None
    heading_level: int | None = None
    section_path: list[str] = field(default_factory=list)


@dataclass
class StructuredDocument:
    metadata: DocumentMetadata
    blocks: list[ContentBlock]

    def to_dict(self) -> dict:
        return {
            "metadata": asdict(self.metadata),
            "block_count": len(self.blocks),
            "blocks": [asdict(block) for block in self.blocks],
        }


def _normalize_text(text: str) -> str:
    cleaned = text.replace("\xa0", " ").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    return cleaned.strip()


def _paragraph_has_bold_text(paragraph: Paragraph) -> bool:
    bold_chars = 0
    total_chars = 0
    for run in paragraph.runs:
        text = run.text.strip()
        if not text:
            continue
        total_chars += len(text)
        if run.bold:
            bold_chars += len(text)
    return total_chars > 0 and bold_chars / total_chars >= 0.55


def _iter_block_items(doc: DocumentObject) -> Iterable[RawBlock]:
    body = doc.element.body
    for child in body.iterchildren():
        if isinstance(child, CT_P):
            paragraph = Paragraph(child, doc)
            text = _normalize_text(paragraph.text)
            if text:
                yield RawBlock(
                    source_kind="paragraph",
                    text=text,
                    style_name=paragraph.style.name if paragraph.style else None,
                    is_bold=_paragraph_has_bold_text(paragraph),
                )
        elif isinstance(child, CT_Tbl):
            table = Table(child, doc)
            for row in table.rows:
                parts: list[str] = []
                for cell in row.cells:
                    cell_text = " ".join(
                        _normalize_text(p.text)
                        for p in cell.paragraphs
                        if _normalize_text(p.text)
                    )
                    if cell_text:
                        parts.append(cell_text)
                text = " | ".join(parts)
                text = _normalize_text(text)
                if text:
                    yield RawBlock(source_kind="table_row", text=text)


def _iter_header_footer_blocks(doc: DocumentObject) -> Iterable[RawBlock]:
    for section in doc.sections:
        for story in (section.header, section.footer):
            for paragraph in story.paragraphs:
                text = _normalize_text(paragraph.text)
                if text:
                    yield RawBlock(
                        source_kind="story_paragraph",
                        text=text,
                        style_name=paragraph.style.name if paragraph.style else None,
                        is_bold=_paragraph_has_bold_text(paragraph),
                    )
            for table in story.tables:
                for row in table.rows:
                    parts: list[str] = []
                    for cell in row.cells:
                        cell_text = " ".join(
                            _normalize_text(p.text)
                            for p in cell.paragraphs
                            if _normalize_text(p.text)
                        )
                        if cell_text:
                            parts.append(cell_text)
                    text = _normalize_text(" | ".join(parts))
                    if text:
                        yield RawBlock(source_kind="story_table_row", text=text)


def _safe_slug(text: str) -> str:
    lowered = text.lower()
    slug = re.sub(r"[^a-zа-яё0-9]+", "-", lowered, flags=re.IGNORECASE)
    slug = slug.strip("-")
    return slug or "document"


def _looks_like_toc_entry(text: str) -> bool:
    if TOC_ENTRY_RE.search(text):
        return True
    if re.match(r"^(?:\d+(?:\.\d+)?|Приложение\s*№?\s*\d+).+\s+\d{1,3}$", text, re.IGNORECASE):
        return True
    return False


def _style_heading_level(block: RawBlock) -> int | None:
    if not block.style_name:
        return None
    match = HEADING_STYLE_RE.match(block.style_name.strip())
    if not match:
        return None
    raw_level = match.group("level")
    return int(raw_level) if raw_level else 1


def _style_is_toc(block: RawBlock) -> bool:
    return bool(block.style_name and TOC_STYLE_RE.match(block.style_name.strip()))


def _parse_numeric_prefix(text: str) -> tuple[str, str, int] | None:
    match = NUMERIC_PREFIX_RE.match(text)
    if not match:
        return None
    number = match.group("number")
    title = match.group("text").strip()
    if title.isdigit() or DATE_RE.fullmatch(number):
        return None
    return number, title, number.count(".") + 1


def _parse_level1(text: str) -> tuple[str, str] | None:
    parsed = _parse_numeric_prefix(text)
    if parsed and parsed[2] == 1:
        return parsed[0], parsed[1]
    return None


def _parse_level2(text: str) -> tuple[str, str] | None:
    parsed = _parse_numeric_prefix(text)
    if parsed and parsed[2] == 2:
        return parsed[0], parsed[1]
    return None


def _parse_numbered_paragraph(text: str) -> tuple[str, str] | None:
    parsed = _parse_numeric_prefix(text)
    if parsed and parsed[2] >= 3:
        return parsed[0], parsed[1]
    return None


def _parse_letter_bullet(text: str) -> tuple[str, str] | None:
    match = LETTER_BULLET_RE.match(text)
    if not match:
        return None
    return match.group("marker"), match.group("text").strip()


def _looks_like_content_start(text: str) -> bool:
    return bool(_parse_numeric_prefix(text) or APPENDIX_RE.match(text) or TOC_TITLE_RE.match(text))


def _looks_like_heading(block: RawBlock, title: str, level: int) -> bool:
    if _style_heading_level(block):
        return True
    if level <= 2:
        return True
    if block.is_bold and len(title) <= 140:
        return True
    if len(title) <= 90 and not re.search(r"[.;:]$", title):
        return True
    return False


def _is_metadata_noise(text: str) -> bool:
    return bool(
        text.upper() == "УТВЕРЖДЕНО"
        or "приказом" in text.lower()
        or INDEX_RE.search(text)
        or EFFECTIVE_DATE_RE.search(text)
        or DECLARED_PAGES_RE.search(text)
        or VERSION_RE.search(text)
        or APPROVAL_ORDER_RE.search(text)
    )


def _extract_metadata(source_path: Path, raw_blocks: list[RawBlock]) -> DocumentMetadata:
    index_code: str | None = None
    approval_date: str | None = None
    approval_order_number: str | None = None
    effective_date: str | None = None
    declared_pages: int | None = None
    version: str | None = None
    organization: str | None = None

    title_lines: list[str] = []
    started_title_capture = False

    meta_blocks = raw_blocks
    for block in meta_blocks:
        text = block.text

        if not index_code:
            match = INDEX_RE.search(text)
            if match:
                index_code = match.group("value").strip()
                started_title_capture = True
                continue

        if started_title_capture:
            if EFFECTIVE_DATE_RE.search(text) or DECLARED_PAGES_RE.search(text):
                started_title_capture = False
            elif TOC_TITLE_RE.match(text):
                started_title_capture = False
            elif _parse_level1(text) or _parse_level2(text) or _parse_numbered_paragraph(text):
                started_title_capture = False
            elif ORG_RE.fullmatch(text):
                organization = organization or text
            elif VERSION_RE.search(text):
                version = version or VERSION_RE.search(text).group("value")
            elif text.upper() == "УТВЕРЖДЕНО":
                continue
            elif "приказом" in text.lower():
                continue
            elif INDEX_RE.search(text):
                continue
            else:
                title_lines.append(text)

        if not approval_date or not approval_order_number:
            match = APPROVAL_ORDER_RE.search(text)
            if match:
                approval_date = approval_date or match.group("date")
                approval_order_number = approval_order_number or match.group("number").strip()

        if not effective_date:
            match = EFFECTIVE_DATE_RE.search(text)
            if match:
                effective_date = match.group("date")

        if declared_pages is None:
            match = DECLARED_PAGES_RE.search(text)
            if match:
                declared_pages = int(match.group("count"))

        if not version:
            match = VERSION_RE.search(text)
            if match:
                version = match.group("value")

        if not organization:
            match = ORG_RE.search(text)
            if match:
                organization = match.group(0)

    cleaned_title_lines = [line for line in title_lines if line]
    display_title = " ".join(cleaned_title_lines).strip() or None
    document_kind = cleaned_title_lines[0] if cleaned_title_lines else None

    if not version and index_code:
        fallback = re.search(r"-(\d+(?:\.\d+)*)$", index_code)
        if fallback:
            version = fallback.group(1)

    doc_id_source = index_code or display_title or source_path.stem
    return DocumentMetadata(
        source_path=str(source_path),
        source_name=source_path.name,
        doc_id=_safe_slug(doc_id_source),
        index_code=index_code,
        display_title=display_title,
        document_kind=document_kind,
        organization=organization,
        version=version,
        approval_date=approval_date,
        approval_order_number=approval_order_number,
        effective_date=effective_date,
        declared_pages=declared_pages,
        title_lines=cleaned_title_lines,
    )


def extract_docx(source_path: str | Path) -> StructuredDocument:
    path = Path(source_path)
    doc = Document(path)
    header_footer_blocks = list(_iter_header_footer_blocks(doc))
    body_blocks = list(_iter_block_items(doc))
    metadata = _extract_metadata(path, header_footer_blocks + body_blocks)

    blocks: list[ContentBlock] = []
    hierarchy: list[tuple[int, str | None, str]] = []
    current_appendix_number: str | None = None
    current_appendix_title: str | None = None
    content_started = False
    in_toc = False
    pending_appendix_title = False

    def context() -> dict:
        section = hierarchy[0] if len(hierarchy) >= 1 else None
        subsection = hierarchy[1] if len(hierarchy) >= 2 else None
        return {
            "section_number": section[1] if section else None,
            "section_title": section[2] if section else None,
            "subsection_number": subsection[1] if subsection else None,
            "subsection_title": subsection[2] if subsection else None,
            "section_path": [number for _, number, _ in hierarchy if number],
        }

    def append_heading(raw_block: RawBlock, number: str | None, title: str, level: int) -> None:
        nonlocal content_started, hierarchy
        hierarchy = [item for item in hierarchy if item[0] < level]
        hierarchy.append((level, number, title))
        content_started = True
        ctx = context()
        block_prefix = number or f"unnumbered-{len(blocks)+1}"
        blocks.append(
            ContentBlock(
                block_id=f"heading-{block_prefix}",
                kind="heading",
                text=title,
                source_kind=raw_block.source_kind,
                heading_level=level,
                **ctx,
            )
        )

    for raw_block in body_blocks:
        text = raw_block.text

        if TOC_TITLE_RE.match(text) or _style_is_toc(raw_block):
            in_toc = True
            continue

        if in_toc:
            if _looks_like_toc_entry(text) or _style_is_toc(raw_block):
                continue
            if _parse_numeric_prefix(text) or APPENDIX_RE.match(text):
                in_toc = False
            else:
                continue

        appendix_match = APPENDIX_RE.match(text)
        if appendix_match:
            current_appendix_number = appendix_match.group("number")
            inline_appendix_title = appendix_match.group("title")
            current_appendix_title = inline_appendix_title.strip() if inline_appendix_title else None
            hierarchy = []
            content_started = True
            pending_appendix_title = current_appendix_title is None
            blocks.append(
                ContentBlock(
                    block_id=f"appendix-{current_appendix_number}",
                    kind="appendix_heading",
                    text=text,
                    source_kind=raw_block.source_kind,
                    appendix_number=current_appendix_number,
                    appendix_title=current_appendix_title,
                )
            )
            continue

        if pending_appendix_title:
            current_appendix_title = text
            pending_appendix_title = False
            blocks.append(
                ContentBlock(
                    block_id=f"appendix-{current_appendix_number}-title",
                    kind="appendix_title",
                    text=text,
                    source_kind=raw_block.source_kind,
                    appendix_number=current_appendix_number,
                    appendix_title=current_appendix_title,
                )
            )
            continue

        if current_appendix_number is not None:
            bullet = _parse_letter_bullet(text)
            if bullet:
                marker, body = bullet
                blocks.append(
                    ContentBlock(
                        block_id=f"appendix-{current_appendix_number}-{len(blocks)+1}",
                        kind="appendix_bullet",
                        text=body,
                        source_kind=raw_block.source_kind,
                        item_marker=marker,
                        appendix_number=current_appendix_number,
                        appendix_title=current_appendix_title,
                    )
                )
                continue

            numbered = _parse_numeric_prefix(text)
            if numbered:
                number, body, _level = numbered
                blocks.append(
                    ContentBlock(
                        block_id=f"appendix-{current_appendix_number}-item-{number}",
                        kind="appendix_numbered_item",
                        text=body,
                        source_kind=raw_block.source_kind,
                        item_number=number,
                        appendix_number=current_appendix_number,
                        appendix_title=current_appendix_title,
                    )
                )
            else:
                blocks.append(
                    ContentBlock(
                        block_id=f"appendix-{current_appendix_number}-{len(blocks)+1}",
                        kind="appendix_paragraph",
                        text=text,
                        source_kind=raw_block.source_kind,
                        appendix_number=current_appendix_number,
                        appendix_title=current_appendix_title,
                    )
                )
            continue

        style_heading_level = _style_heading_level(raw_block)
        numbered_prefix = _parse_numeric_prefix(text)

        if style_heading_level:
            if numbered_prefix:
                number, title, numeric_level = numbered_prefix
                append_heading(raw_block, number, title, numeric_level)
            else:
                append_heading(raw_block, None, text, style_heading_level)
            continue

        if numbered_prefix:
            number, body, numeric_level = numbered_prefix
            if _looks_like_heading(raw_block, body, numeric_level):
                append_heading(raw_block, number, body, numeric_level)
                continue
            content_started = True
            blocks.append(
                ContentBlock(
                    block_id=f"item-{number}",
                    kind="numbered_paragraph",
                    text=body,
                    source_kind=raw_block.source_kind,
                    item_number=number,
                    **context(),
                )
            )
            continue

        bullet = _parse_letter_bullet(text)
        if bullet and content_started:
            marker, body = bullet
            blocks.append(
                ContentBlock(
                    block_id=f"bullet-{len(blocks)+1}",
                    kind="letter_bullet",
                    text=body,
                    source_kind=raw_block.source_kind,
                    item_marker=marker,
                    **context(),
                )
            )
            continue

        if content_started:
            blocks.append(
                ContentBlock(
                    block_id=f"paragraph-{len(blocks)+1}",
                    kind="paragraph",
                    text=text,
                    source_kind=raw_block.source_kind,
                    **context(),
                )
            )
        elif not _is_metadata_noise(text):
            blocks.append(
                ContentBlock(
                    block_id=f"front-matter-{len(blocks)+1}",
                    kind="front_matter",
                    text=text,
                    source_kind=raw_block.source_kind,
                )
            )

    return StructuredDocument(metadata=metadata, blocks=blocks)


def extract_docx_text(source_path: str | Path) -> str:
    doc = Document(Path(source_path))
    return "\n\n".join(block.text for block in _iter_block_items(doc) if block.text).strip()


def extract_many(paths: Iterable[str | Path]) -> list[StructuredDocument]:
    return [extract_docx(path) for path in paths]


def write_extraction(doc: StructuredDocument, destination: str | Path) -> Path:
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(doc.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path
