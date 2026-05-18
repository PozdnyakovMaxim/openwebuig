from __future__ import annotations

import argparse
import json
from pathlib import Path


RECOMMENDED_METADATA_FIELDS = ("index_code", "display_title", "version")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate structured JSON produced by step1_extract_corpus.py."
    )
    parser.add_argument("--input-dir", required=True, help="Directory with manifest.json and extracted JSON files.")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_outputs = {Path(item["output_path"]).resolve() for item in manifest}
    existing_outputs = {path.resolve() for path in input_dir.glob("*.json") if path.name != "manifest.json"}

    errors: list[str] = []
    warnings: list[str] = []

    stale_outputs = sorted(existing_outputs - expected_outputs)
    if stale_outputs:
        warnings.append("stale JSON files not referenced by manifest: " + ", ".join(path.name for path in stale_outputs))

    for item in manifest:
        output_path = Path(item["output_path"]).resolve()
        if not output_path.exists():
            errors.append(f"missing extracted JSON for {item['source_name']}: {output_path}")
            continue

        data = json.loads(output_path.read_text(encoding="utf-8"))
        metadata = data.get("metadata") or {}
        blocks = data.get("blocks") or []

        if data.get("block_count") != len(blocks):
            errors.append(f"{output_path.name}: block_count does not match blocks length")
        if not blocks:
            errors.append(f"{output_path.name}: no content blocks extracted")

        missing_recommended = [field for field in RECOMMENDED_METADATA_FIELDS if not metadata.get(field)]
        if missing_recommended:
            warnings.append(f"{output_path.name}: missing recommended metadata fields: {', '.join(missing_recommended)}")

        contextual_blocks = [
            block
            for block in blocks
            if block.get("kind") not in {"front_matter", "appendix_heading", "appendix_title", "appendix_paragraph", "appendix_numbered_item", "appendix_bullet"}
        ]
        if contextual_blocks:
            without_context = [block for block in contextual_blocks if not block.get("section_path")]
            ratio = len(without_context) / len(contextual_blocks)
            if ratio > 0.25:
                warnings.append(f"{output_path.name}: many main blocks have no section_path ({len(without_context)}/{len(contextual_blocks)})")

    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")

    print(f"Validated {len(manifest)} extracted documents from {input_dir}")
    if errors or (args.strict and warnings):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
