from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate chunk JSON produced by step2_chunk_corpus.py.")
    parser.add_argument("--input-dir", required=True, help="Directory with chunk manifest.json and *.chunks.json files.")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []
    seen_chunk_ids: set[str] = set()

    for item in manifest:
        path = Path(item["chunked_path"]).resolve()
        if not path.exists():
            errors.append(f"missing chunked file: {path}")
            continue

        data = json.loads(path.read_text(encoding="utf-8"))
        chunks = data.get("chunks") or []
        if data.get("chunk_count") != len(chunks):
            errors.append(f"{path.name}: chunk_count does not match chunks length")
        if not chunks:
            errors.append(f"{path.name}: no chunks")

        metadata_chunks = [chunk for chunk in chunks if chunk.get("chunk_type") == "metadata"]
        if not metadata_chunks:
            warnings.append(f"{path.name}: no metadata chunk")

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                errors.append(f"{path.name}: chunk without chunk_id")
            elif chunk_id in seen_chunk_ids:
                errors.append(f"{path.name}: duplicate chunk_id {chunk_id}")
            else:
                seen_chunk_ids.add(chunk_id)

            if not chunk.get("raw_text"):
                errors.append(f"{path.name}: {chunk_id} has empty raw_text")
            if not chunk.get("searchable_text"):
                errors.append(f"{path.name}: {chunk_id} has empty searchable_text")
            if not chunk.get("citation_label"):
                errors.append(f"{path.name}: {chunk_id} has empty citation_label")
            if not chunk.get("block_ids"):
                errors.append(f"{path.name}: {chunk_id} has no block_ids")
            if chunk.get("chunk_type") not in {"metadata", "appendix_item", "appendix_list", "appendix_text"}:
                if not chunk.get("section_path") and chunk.get("chunk_type") != "section_text":
                    warnings.append(f"{path.name}: {chunk_id} has no section_path")

    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    print(f"Validated {len(manifest)} chunked documents from {input_dir}")

    if errors or (args.strict and warnings):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
