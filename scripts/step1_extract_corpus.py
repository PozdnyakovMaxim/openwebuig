from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from document_search.extractor import extract_docx, write_extraction


def resolve_inputs(patterns: list[str]) -> list[Path]:
    resolved: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path(match) for match in glob.glob(pattern))
        if matches:
            resolved.extend(matches)
            continue
        path = Path(pattern)
        if path.exists():
            resolved.append(path)
    unique = []
    seen: set[Path] = set()
    for path in resolved:
        absolute = path.resolve()
        if absolute not in seen and absolute.suffix.lower() == ".docx":
            seen.add(absolute)
            unique.append(absolute)
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 1: extract DOCX documents into structured JSON."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Glob pattern or explicit DOCX path. Pass multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where extracted JSON and manifest.json will be written.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not remove old JSON files from the output directory before extraction.",
    )
    args = parser.parse_args()

    input_paths = resolve_inputs(args.input)
    if not input_paths:
        raise SystemExit("No DOCX files matched the provided --input patterns.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_clean:
        for stale_json in output_dir.glob("*.json"):
            stale_json.unlink()

    manifest: list[dict] = []
    for source_path in input_paths:
        extracted = extract_docx(source_path)
        output_path = output_dir / f"{extracted.metadata.doc_id}.json"
        write_extraction(extracted, output_path)
        manifest.append(
            {
                "source_name": extracted.metadata.source_name,
                "source_path": extracted.metadata.source_path,
                "doc_id": extracted.metadata.doc_id,
                "index_code": extracted.metadata.index_code,
                "display_title": extracted.metadata.display_title,
                "version": extracted.metadata.version,
                "blocks": len(extracted.blocks),
                "output_path": str(output_path),
            }
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Extracted {len(manifest)} documents to {output_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
