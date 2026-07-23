from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import posixpath
import re
from typing import Any, BinaryIO
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile


WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+", re.UNICODE)
WORDPROCESSINGML_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
MARKUP_COMPATIBILITY_NS = "http://schemas.openxmlformats.org/markup-compatibility/2006"
W = f"{{{WORDPROCESSINGML_NS}}}"
MC = f"{{{MARKUP_COMPATIBILITY_NS}}}"


@dataclass(frozen=True)
class SourceTextSegment:
    part: str
    story: str
    text: str
    style: str = ""
    location: str = "paragraph"
    has_dynamic_page_field: bool = False


def source_ooxml_inventory(source: str | Path | bytes) -> dict[str, Any]:
    """Return canonical visible WordprocessingML text from a DOCX package."""

    segments: list[SourceTextSegment] = []
    archive_source: str | Path | BinaryIO
    archive_source = BytesIO(source) if isinstance(source, bytes) else source
    try:
        with ZipFile(archive_source) as archive:
            names = set(archive.namelist())
            if "word/document.xml" not in names:
                raise ValueError("word/document.xml is missing")
            document_root = _read_ooxml_part(archive, "word/document.xml")
            segments.extend(
                _ooxml_paragraph_segments(document_root, "word/document.xml", "body")
            )
            for part_name, story, note_ids in _referenced_story_parts(
                archive,
                names,
                document_root,
            ):
                root = _read_ooxml_part(archive, part_name)
                segments.extend(
                    _ooxml_paragraph_segments(
                        root,
                        part_name,
                        story,
                        note_ids=note_ids,
                    )
                )
    except (BadZipFile, KeyError, OSError, ValueError) as exc:
        raise ValueError(f"cannot inventory DOCX OOXML: {exc}") from exc

    required, ignored = partition_source_segments(segments)
    story_counts = Counter(segment.story for segment in required)
    location_counts = Counter(segment.location for segment in required)
    return {
        "segments": required,
        "ignored_segments": ignored,
        "story_counts": dict(sorted(story_counts.items())),
        "location_counts": dict(sorted(location_counts.items())),
    }


def _read_ooxml_part(archive: ZipFile, part_name: str) -> ElementTree.Element:
    try:
        return ElementTree.fromstring(archive.read(part_name))
    except ElementTree.ParseError as exc:
        raise ValueError(f"invalid XML in {part_name}: {exc}") from exc


def _referenced_story_parts(
    archive: ZipFile,
    names: set[str],
    document_root: ElementTree.Element,
) -> list[tuple[str, str, set[str] | None]]:
    relationships_name = "word/_rels/document.xml.rels"
    if relationships_name not in names:
        return []
    relationships_root = _read_ooxml_part(archive, relationships_name)
    relationships: dict[str, tuple[str, str]] = {}
    for relationship in relationships_root:
        if _xml_local_name(relationship.tag) != "Relationship":
            continue
        relationship_id = str(relationship.get("Id") or "")
        relationship_type = str(relationship.get("Type") or "").rsplit("/", 1)[-1]
        target = str(relationship.get("Target") or "")
        if (
            not relationship_id
            or relationship_type not in {"header", "footer", "footnotes", "endnotes"}
            or not target
            or str(relationship.get("TargetMode") or "").casefold() == "external"
        ):
            continue
        part_name = _resolve_ooxml_target("word/document.xml", target)
        if part_name not in names:
            raise ValueError(
                f"{relationships_name} references missing OOXML part {part_name}"
            )
        relationships[relationship_id] = (part_name, relationship_type)

    referenced_headers = _visible_relationship_ids(
        document_root,
        {f"{W}headerReference", f"{W}footerReference"},
    )
    note_ids = {
        "footnotes": _visible_note_ids(document_root, f"{W}footnoteReference"),
        "endnotes": _visible_note_ids(document_root, f"{W}endnoteReference"),
    }
    selected: dict[str, tuple[str, str, set[str] | None]] = {}
    for relationship_id, (part_name, relationship_type) in relationships.items():
        if relationship_type in {"header", "footer"}:
            if relationship_id not in referenced_headers:
                continue
            selected[part_name] = (part_name, relationship_type, None)
            continue
        referenced_notes = note_ids[relationship_type]
        if referenced_notes:
            selected[part_name] = (
                part_name,
                relationship_type.removesuffix("s"),
                referenced_notes,
            )
    return [selected[name] for name in sorted(selected)]


def _resolve_ooxml_target(source_part: str, target: str) -> str:
    if target.startswith("/"):
        resolved = posixpath.normpath(target.lstrip("/"))
    else:
        resolved = posixpath.normpath(
            posixpath.join(posixpath.dirname(source_part), target)
        )
    if resolved in {"", ".", ".."} or resolved.startswith("../"):
        raise ValueError(f"unsafe OOXML relationship target: {target!r}")
    return resolved


def _visible_relationship_ids(
    root: ElementTree.Element,
    tags: set[str],
) -> set[str]:
    values: set[str] = set()

    def visit(node: ElementTree.Element) -> None:
        if node.tag in {f"{W}del", f"{W}moveFrom"}:
            return
        if node.tag == f"{W}r" and _run_is_hidden(node):
            return
        if node.tag in tags:
            for attribute, value in node.attrib.items():
                if _xml_local_name(attribute) == "id" and value:
                    values.add(value)
                    break
        for child in _visible_xml_children(node):
            visit(child)

    visit(root)
    return values


def _visible_note_ids(root: ElementTree.Element, tag: str) -> set[str]:
    values: set[str] = set()

    def visit(node: ElementTree.Element) -> None:
        if node.tag in {f"{W}del", f"{W}moveFrom"}:
            return
        if node.tag == f"{W}r" and _run_is_hidden(node):
            return
        if node.tag == tag:
            value = node.get(f"{W}id")
            if value:
                values.add(value)
        for child in _visible_xml_children(node):
            visit(child)

    visit(root)
    return values


def _xml_local_name(value: str) -> str:
    return value.rsplit("}", 1)[-1]


def _ooxml_paragraph_segments(
    root: ElementTree.Element,
    part_name: str,
    story: str,
    *,
    note_ids: set[str] | None = None,
) -> list[SourceTextSegment]:
    segments: list[SourceTextSegment] = []

    def visit(
        node: ElementTree.Element,
        *,
        in_textbox: bool = False,
        in_table: bool = False,
    ) -> None:
        if node.tag in {f"{W}footnote", f"{W}endnote"} and node.get(f"{W}type"):
            return
        if (
            note_ids is not None
            and node.tag in {f"{W}footnote", f"{W}endnote"}
            and node.get(f"{W}id") not in note_ids
        ):
            return
        if node.tag == f"{MC}AlternateContent":
            branch = _alternate_content_branch(node)
            if branch is not None:
                visit(branch, in_textbox=in_textbox, in_table=in_table)
            return

        nested_textbox = in_textbox or node.tag == f"{W}txbxContent"
        nested_table = in_table or node.tag == f"{W}tbl"
        if node.tag == f"{W}p":
            text = _visible_ooxml_paragraph_text(node)
            if text:
                segments.append(
                    SourceTextSegment(
                        part=part_name,
                        story=story,
                        text=text,
                        style=_ooxml_paragraph_style(node),
                        location=(
                            "textbox"
                            if nested_textbox
                            else "table"
                            if nested_table
                            else "paragraph"
                        ),
                        has_dynamic_page_field=_has_dynamic_page_field(node),
                    )
                )

        for child in _visible_xml_children(node):
            visit(child, in_textbox=nested_textbox, in_table=nested_table)

    visit(root)
    return segments


def _visible_ooxml_paragraph_text(paragraph: ElementTree.Element) -> str:
    parts: list[str] = []

    def collect(node: ElementTree.Element, *, root: bool = False) -> None:
        if not root and node.tag == f"{W}p":
            return
        if node.tag in {f"{W}del", f"{W}moveFrom"}:
            return
        if node.tag == f"{W}r" and _run_is_hidden(node):
            return
        if node.tag == f"{W}t":
            if node.text:
                parts.append(node.text)
            return
        if node.tag == f"{W}tab":
            parts.append(" ")
            return
        if node.tag in {f"{W}br", f"{W}cr"}:
            parts.append("\n")
            return
        if node.tag == f"{MC}AlternateContent":
            branch = _alternate_content_branch(node)
            if branch is not None:
                collect(branch)
            return
        for child in _visible_xml_children(node):
            collect(child)

    collect(paragraph, root=True)
    return _normalize_source_text("".join(parts))


def _visible_xml_children(node: ElementTree.Element) -> list[ElementTree.Element]:
    if node.tag != f"{MC}AlternateContent":
        return list(node)
    branch = _alternate_content_branch(node)
    return [branch] if branch is not None else []


def _alternate_content_branch(node: ElementTree.Element) -> ElementTree.Element | None:
    choice = node.find(f"{MC}Choice")
    return choice if choice is not None else node.find(f"{MC}Fallback")


def _run_is_hidden(run: ElementTree.Element) -> bool:
    properties = run.find(f"{W}rPr")
    if properties is None:
        return False
    return any(
        _ooxml_boolean_is_on(properties.find(f"{W}{name}"))
        for name in ("vanish", "webHidden")
    )


def _ooxml_boolean_is_on(element: ElementTree.Element | None) -> bool:
    if element is None:
        return False
    value = str(element.get(f"{W}val") or "true").strip().casefold()
    return value not in {"0", "false", "off", "no"}


def _ooxml_paragraph_style(paragraph: ElementTree.Element) -> str:
    properties = paragraph.find(f"{W}pPr")
    style = properties.find(f"{W}pStyle") if properties is not None else None
    return str(style.get(f"{W}val") or "") if style is not None else ""


def _has_dynamic_page_field(paragraph: ElementTree.Element) -> bool:
    instructions = " ".join(
        str(node.text or "") for node in paragraph.iter(f"{W}instrText")
    ).upper()
    return bool(re.search(r"\b(?:PAGE|NUMPAGES|SECTIONPAGES)\b", instructions))


def partition_source_segments(
    segments: list[SourceTextSegment],
) -> tuple[list[SourceTextSegment], list[SourceTextSegment]]:
    required: list[SourceTextSegment] = []
    ignored: list[SourceTextSegment] = []
    toc_parts: dict[str, bool] = defaultdict(bool)

    for segment in segments:
        normalized = _normalize_source_text(segment.text)
        style = segment.style.casefold().replace(" ", "")
        is_toc_style = style.startswith(("toc", "contents", "оглавлен", "содержан"))
        if segment.story == "body":
            if normalized.casefold() in {"содержание", "оглавление"}:
                toc_parts[segment.part] = True
                ignored.append(segment)
                continue
            if is_toc_style:
                toc_parts[segment.part] = True
                ignored.append(segment)
                continue
            if toc_parts[segment.part] and _looks_like_ooxml_toc_entry(normalized):
                ignored.append(segment)
                continue
            toc_parts[segment.part] = False

        if segment.story in {"header", "footer"}:
            if _is_page_counter_segment(segment, normalized):
                ignored.append(segment)
                continue
        required.append(segment)
    return required, ignored


def _looks_like_literal_page_number(text: str) -> bool:
    return bool(
        re.fullmatch(
            r"(?:(?:стр(?:аница)?\.?|page)\s*(?:№\s*)?\d+"
            r"(?:\s*(?:из|of|/)\s*\d+)?|\d+\s*/\s*\d+)",
            text,
            re.IGNORECASE,
        )
    )


def _is_page_counter_segment(segment: SourceTextSegment, text: str) -> bool:
    tokens = _tokens(text)
    if tokens and all(token.isdigit() for token in tokens):
        return True
    if _looks_like_literal_page_number(text):
        return True
    if not segment.has_dynamic_page_field:
        return False
    page_words = {"стр", "страница", "page", "из", "of"}
    return bool(tokens) and all(
        token.isdigit()
        or token in page_words
        or bool(re.fullmatch(r"[ivxlcdm]+", token, re.IGNORECASE))
        for token in tokens
    )


def _looks_like_ooxml_toc_entry(text: str) -> bool:
    if re.search(r"(?:\.{2,}|\s{2,}|\t)\s*\d{1,4}$", text):
        return True
    return bool(
        re.match(
            r"^(?:\d+(?:\.\d+)*|(?:Приложение|Appendix)\s*№?\s*\d+).+\s+\d{1,4}$",
            text,
            re.IGNORECASE,
        )
    )


def _normalize_source_text(text: str) -> str:
    cleaned = text.replace("\xa0", " ").replace("\r", "\n")
    return re.sub(r"\s+", " ", cleaned).strip()


def _tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in WORD_RE.finditer(text)]
