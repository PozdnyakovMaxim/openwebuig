from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from document_search.provider_api import make_chat
from document_search.settings import load_env_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure RAG API or provider chat latency.")
    parser.add_argument("query", help="Question used for every run.")
    parser.add_argument("--target", choices=("rag", "provider"), default="rag")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rag-url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument(
        "--api-key",
        default=None,
        help="RAG API key. Defaults to OPENAI_COMPAT_API_KEY from the environment/.env.",
    )
    args = parser.parse_args()
    load_env_file()
    api_key = args.api_key or os.getenv("OPENAI_COMPAT_API_KEY") or ""
    if args.target == "rag" and not api_key:
        parser.error("--api-key or OPENAI_COMPAT_API_KEY is required for the RAG target")

    for _ in range(args.warmup):
        if args.target == "rag":
            _call_rag(args.rag_url, api_key, args.query)
        else:
            _call_provider(args.query)

    samples: list[float] = []
    for run_number in range(1, args.runs + 1):
        started = time.perf_counter()
        if args.target == "rag":
            details = _call_rag(args.rag_url, api_key, args.query)
        else:
            details = _call_provider(args.query)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        samples.append(elapsed_ms)
        print(f"run={run_number} total_ms={elapsed_ms:.2f}{details}")

    ordered = sorted(samples)
    p95_index = max(0, min(len(ordered) - 1, int(len(ordered) * 0.95 + 0.9999) - 1))
    print(
        "summary "
        f"runs={len(samples)} min_ms={min(samples):.2f} "
        f"avg_ms={statistics.mean(samples):.2f} p95_ms={ordered[p95_index]:.2f} "
        f"max_ms={max(samples):.2f}"
    )
    return 0


def _call_rag(url: str, api_key: str, query: str) -> str:
    payload = {
        "model": "document-search-rag",
        "messages": [{"role": "user", "content": query}],
        "temperature": 0,
        "stream": False,
    }
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urlopen(request, timeout=300) as response:
        data = json.loads(response.read().decode("utf-8"))
        metrics = data.get("rag_metrics") or {}
        timings = metrics.get("timings_ms") or {}
        route = metrics.get("route") or response.headers.get("X-RAG-Route") or "unknown"
    fields = [f" route={route}"]
    for name in ("routing", "embedding", "search", "database", "generation"):
        if name in timings:
            fields.append(f" {name}_ms={float(timings[name]):.2f}")
    return "".join(fields)


def _call_provider(query: str) -> str:
    chat = make_chat()
    chat.complete(
        [
            {"role": "system", "content": "Ответь одним коротким предложением."},
            {"role": "user", "content": query},
        ],
        temperature=0,
    )
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
