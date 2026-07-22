from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path
import tempfile
from urllib.parse import unquote
from zipfile import BadZipFile, ZipFile

from .extractor import extract_docx_text


DOCX_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".xml", ".yaml", ".yml"}
DOCX_MAX_MEMBER_COUNT = 10_000
DOCX_MAX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
DOCX_MAX_COMPRESSION_RATIO = 500.0
DOCX_COMPRESSION_RATIO_MIN_BYTES = 1024 * 1024


class ExternalDocumentError(ValueError):
    pass


class DocumentTooLargeError(ExternalDocumentError):
    pass


class UnsupportedDocumentError(ExternalDocumentError):
    pass


class InvalidDocumentError(ExternalDocumentError):
    pass


def process_external_document(
    payload: bytes,
    *,
    filename: str,
    content_type: str = "",
    max_bytes: int = 64 * 1024 * 1024,
) -> dict[str, object]:
    """Return the response expected by Open WebUI's ExternalDocumentLoader."""

    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    if not payload:
        raise InvalidDocumentError("Uploaded document is empty")
    if len(payload) > max_bytes:
        raise DocumentTooLargeError(
            f"Uploaded document exceeds the {max_bytes}-byte limit"
        )

    safe_name = _safe_filename(filename)
    suffix = Path(safe_name).suffix.casefold()
    normalized_content_type = content_type.partition(";")[0].strip().casefold()
    is_docx = suffix == ".docx" or normalized_content_type == DOCX_CONTENT_TYPE

    if is_docx:
        _validate_docx(payload)
        with tempfile.TemporaryDirectory(prefix="document-search-upload-") as temp_dir:
            source_path = Path(temp_dir) / "uploaded.docx"
            source_path.write_bytes(payload)
            try:
                text = extract_docx_text(source_path)
            except Exception as exc:
                raise InvalidDocumentError("Uploaded DOCX could not be parsed") from exc
        parser = "document_search.docx_ooxml"
        numbering_preserved = True
    elif suffix in TEXT_EXTENSIONS or normalized_content_type.startswith("text/"):
        try:
            text = payload.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise InvalidDocumentError("Text document must use UTF-8 encoding") from exc
        parser = "document_search.utf8_text"
        numbering_preserved = False
    else:
        raise UnsupportedDocumentError(
            "This external loader supports DOCX and UTF-8 text documents"
        )

    text = text.strip()
    if not text:
        raise InvalidDocumentError("Document contains no extractable text")
    return {
        "page_content": text,
        "metadata": {
            "source": safe_name,
            "file_name": safe_name,
            "content_type": normalized_content_type or "application/octet-stream",
            "source_sha256": sha256(payload).hexdigest(),
            "parser": parser,
            "word_numbering_preserved": numbering_preserved,
        },
    }


def _safe_filename(value: str) -> str:
    decoded = unquote(str(value or "document.docx")).replace("\x00", "")
    name = Path(decoded).name.strip()
    return name or "document.docx"


def _validate_docx(
    payload: bytes,
    *,
    max_member_count: int = DOCX_MAX_MEMBER_COUNT,
    max_uncompressed_bytes: int = DOCX_MAX_UNCOMPRESSED_BYTES,
    max_compression_ratio: float = DOCX_MAX_COMPRESSION_RATIO,
) -> None:
    if max_member_count <= 0 or max_uncompressed_bytes <= 0 or max_compression_ratio <= 0:
        raise ValueError("DOCX archive limits must be positive")
    try:
        with ZipFile(BytesIO(payload)) as archive:
            members = archive.infolist()
            if len(members) > max_member_count:
                raise InvalidDocumentError(
                    f"Uploaded DOCX contains too many ZIP members ({len(members)})"
                )

            total_uncompressed = 0
            for member in members:
                if member.is_dir():
                    continue
                total_uncompressed += member.file_size
                if total_uncompressed > max_uncompressed_bytes:
                    raise InvalidDocumentError(
                        "Uploaded DOCX expands beyond the permitted uncompressed size"
                    )
                if member.file_size < DOCX_COMPRESSION_RATIO_MIN_BYTES:
                    continue
                if member.compress_size <= 0:
                    raise InvalidDocumentError(
                        "Uploaded DOCX contains an invalid compressed ZIP member"
                    )
                ratio = member.file_size / member.compress_size
                if ratio > max_compression_ratio:
                    raise InvalidDocumentError(
                        "Uploaded DOCX contains a suspicious ZIP compression ratio"
                    )
            names = {member.filename for member in members}
    except BadZipFile as exc:
        raise InvalidDocumentError("Uploaded DOCX is not a valid ZIP package") from exc
    if "word/document.xml" not in names or "[Content_Types].xml" not in names:
        raise InvalidDocumentError("Uploaded file is not a valid DOCX document")
