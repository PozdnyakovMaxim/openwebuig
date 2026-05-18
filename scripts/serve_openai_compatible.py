from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from document_search.settings import load_env_file


def main() -> int:
    load_env_file()
    parser = argparse.ArgumentParser(description="Serve the RAG system as an OpenAI-compatible API.")
    parser.add_argument("--host", default=os.getenv("OPENAI_COMPAT_HOST") or "0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.getenv("OPENAI_COMPAT_PORT") or "8000"))
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Missing dependency uvicorn. Run `uv sync`.") from exc

    uvicorn.run("document_search.openai_compatible:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
