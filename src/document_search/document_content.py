from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

from .extractor import extract_docx_text


def load_source_document_text(document: dict[str, Any]) -> str | None:
    source_path = find_source_document(document)
    if source_path is None:
        return None
    return extract_docx_text(source_path)


def find_source_document(
    document: dict[str, Any],
    *,
    roots: Iterable[Path] | None = None,
) -> Path | None:
    metadata = document.get("metadata")
    if isinstance(metadata, dict):
        stored_path = Path(str(metadata.get("source_path") or "")).expanduser()
        if stored_path.is_file() and stored_path.suffix.lower() == ".docx":
            return stored_path

    source_name = str(document.get("source_name") or "").strip()
    if not source_name:
        return None

    search_roots = list(roots) if roots is not None else _document_roots()
    for root in search_roots:
        direct = root.expanduser() / source_name
        if direct.is_file():
            return direct

    expected_name = Path(source_name).name.casefold()
    for root in search_roots:
        expanded = root.expanduser()
        if not expanded.is_dir():
            continue
        for candidate in expanded.rglob("*.docx"):
            if candidate.name.casefold() == expected_name:
                return candidate
    return None


def _document_roots() -> list[Path]:
    roots: list[Path] = []
    configured = os.getenv("RAG_DOCUMENT_ROOTS") or ""
    for value in configured.split(os.pathsep):
        if value.strip():
            roots.append(Path(value.strip()))
    roots.extend(
        [
            Path.home() / "rag_template" / "docs",
            Path.home() / "openwebuig" / "docs",
            Path.cwd() / "docs",
        ]
    )
    return roots
