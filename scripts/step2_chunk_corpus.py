from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from document_search.chunker import DEFAULT_MAX_CHARS, chunk_many


def resolve_extracted_inputs(input_dir: Path) -> list[Path]:
    manifest_path = input_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        paths: list[Path] = []
        for item in manifest:
            path = Path(item["output_path"])
            if path.exists():
                paths.append(path.resolve())
                continue
            fallback = input_dir / path.name
            if fallback.exists():
                paths.append(fallback.resolve())
        return paths
    return sorted(path for path in input_dir.glob("*.json") if path.name != "manifest.json")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 2: build citation-ready chunks from extracted JSON."
    )
    parser.add_argument("--input-dir", required=True, help="Directory produced by step1_extract_corpus.py.")
    parser.add_argument("--output-dir", required=True, help="Directory where chunk JSON files will be written.")
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS, help="Maximum raw_text length per chunk.")
    parser.add_argument("--no-clean", action="store_true", help="Do not remove old chunk JSON files from output dir.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_clean:
        for stale in output_dir.glob("*.json"):
            stale.unlink()

    inputs = resolve_extracted_inputs(input_dir)
    if not inputs:
        raise SystemExit(f"No extracted JSON files found in {input_dir}")

    manifest = chunk_many(inputs, output_dir, max_chars=args.max_chars)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Chunked {len(manifest)} documents to {output_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
