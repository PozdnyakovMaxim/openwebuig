from __future__ import annotations

import argparse
import ipaddress
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from document_search.settings import load_env_file


def validate_bind_auth(host: str, api_key: str) -> None:
    normalized_host = host.strip().strip("[]")
    is_loopback = normalized_host.casefold() == "localhost"
    if not is_loopback:
        try:
            is_loopback = ipaddress.ip_address(normalized_host).is_loopback
        except ValueError:
            is_loopback = False
    if is_loopback:
        return
    normalized_key = api_key.strip()
    if len(normalized_key) < 16 or normalized_key.casefold().startswith(
        ("replace_", "change_me")
    ):
        raise ValueError(
            "OPENAI_COMPAT_API_KEY must be a non-placeholder secret of at least "
            "16 characters when binding beyond loopback"
        )


def main() -> int:
    load_env_file()
    parser = argparse.ArgumentParser(description="Serve the RAG system as an OpenAI-compatible API.")
    parser.add_argument("--host", default=os.getenv("OPENAI_COMPAT_HOST") or "0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.getenv("OPENAI_COMPAT_PORT") or "8000"))
    args = parser.parse_args()
    try:
        validate_bind_auth(args.host, os.getenv("OPENAI_COMPAT_API_KEY") or "")
    except ValueError as exc:
        parser.error(str(exc))

    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Missing dependency uvicorn. Run `uv sync`.") from exc

    uvicorn.run("document_search.openai_compatible:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
