#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


EXPECTED_LIST_FIELDS = (
    "expected_source_substrings",
    "expected_title_substrings",
    "expected_answer_substrings",
    "forbidden_substrings",
)


@dataclass(frozen=True)
class ApiObservation:
    answer: str
    route: str | None
    sources: list[str]
    rag_metrics: dict[str, Any]
    latency_ms: float

    @property
    def server_total_ms(self) -> float | None:
        timings = self.rag_metrics.get("timings_ms")
        if not isinstance(timings, dict):
            return None
        value = timings.get("total")
        if not isinstance(value, int | float) or isinstance(value, bool):
            return None
        return round(float(value), 2)


def parse_sources(answer: str) -> list[str]:
    """Extract source labels from the final Russian or English Sources block."""
    lines = answer.splitlines()
    header_index: int | None = None
    for index, line in enumerate(lines):
        normalized = line.strip().casefold()
        if normalized in {"источники:", "sources:"}:
            header_index = index

    if header_index is None:
        return []

    sources: list[str] = []
    for line in lines[header_index + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("[") or "]" not in stripped:
            if sources:
                break
            continue
        closing = stripped.find("]")
        if not stripped[1:closing].isdigit():
            continue
        label = stripped[closing + 1 :].strip()
        if label:
            sources.append(label)
    return sources


def parse_api_response(payload: dict[str, Any], *, latency_ms: float) -> ApiObservation:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("API response has no choices")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError("API response contains an invalid choice")
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("API response choice has no message")
    answer = message.get("content")
    if not isinstance(answer, str):
        raise ValueError("API response message content is not text")

    metrics = payload.get("rag_metrics")
    rag_metrics = metrics if isinstance(metrics, dict) else {}
    route_value = rag_metrics.get("route")
    route = str(route_value).strip() if route_value not in (None, "") else None
    return ApiObservation(
        answer=answer,
        route=route,
        sources=parse_sources(answer),
        rag_metrics=rag_metrics,
        latency_ms=round(float(latency_ms), 2),
    )


def build_messages(case: dict[str, Any]) -> list[dict[str, str]]:
    history = case.get("history") or []
    messages = [
        {"role": str(item["role"]), "content": str(item["content"])}
        for item in history
    ]
    messages.append({"role": "user", "content": str(case["query"])})
    return messages


def evaluate_case(case: dict[str, Any], observation: ApiObservation) -> dict[str, Any]:
    case_sensitive = bool(case.get("case_sensitive", False))
    failures: list[str] = []
    expected_route = case.get("expected_route")
    route_ok: bool | None = None
    if expected_route:
        route_ok = _equal_text(str(expected_route), observation.route or "", case_sensitive)
        if not route_ok:
            failures.append(
                f"route: expected {expected_route!r}, got {observation.route or '<missing>'!r}"
            )

    expected_sources = list(case.get("expected_source_substrings") or [])
    matched_sources = [
        value
        for value in expected_sources
        if _contains_in_any(value, observation.sources, case_sensitive)
    ]
    for value in expected_sources:
        if value not in matched_sources:
            failures.append(f"source substring not found: {value!r}")

    expected_titles = list(case.get("expected_title_substrings") or [])
    matched_titles = [
        value
        for value in expected_titles
        if _contains(value, observation.answer, case_sensitive)
    ]
    for value in expected_titles:
        if value not in matched_titles:
            failures.append(f"title substring not found in answer: {value!r}")

    expected_answer = list(case.get("expected_answer_substrings") or [])
    matched_answer = [
        value
        for value in expected_answer
        if _contains(value, observation.answer, case_sensitive)
    ]
    for value in expected_answer:
        if value not in matched_answer:
            failures.append(f"answer substring not found: {value!r}")

    forbidden = list(case.get("forbidden_substrings") or [])
    found_forbidden = [
        value
        for value in forbidden
        if _contains(value, observation.answer, case_sensitive)
    ]
    for value in found_forbidden:
        failures.append(f"forbidden substring found: {value!r}")

    max_latency = case.get("max_latency_ms")
    latency_ok: bool | None = None
    if max_latency is not None:
        latency_ok = observation.latency_ms <= float(max_latency)
        if not latency_ok:
            failures.append(
                f"latency: expected <= {float(max_latency):.2f} ms, got {observation.latency_ms:.2f} ms"
            )

    return {
        "id": case["id"],
        "query": case["query"],
        "passed": not failures,
        "failures": failures,
        "route": observation.route,
        "sources": observation.sources,
        "latency_ms": observation.latency_ms,
        "server_total_ms": observation.server_total_ms,
        "answer_preview": _preview(observation.answer),
        "rag_metrics": observation.rag_metrics,
        "checks": {
            "route": route_ok,
            "sources": {
                "matched": len(matched_sources),
                "expected": len(expected_sources),
            },
            "titles": {
                "matched": len(matched_titles),
                "expected": len(expected_titles),
            },
            "answer_substrings": {
                "matched": len(matched_answer),
                "expected": len(expected_answer),
            },
            "forbidden_substrings": {
                "found": len(found_forbidden),
                "expected_absent": len(forbidden),
            },
            "latency": latency_ok,
        },
    }


def evaluation_error_result(
    case: dict[str, Any],
    error: Exception | str,
    *,
    latency_ms: float | None = None,
) -> dict[str, Any]:
    expected_sources = len(case.get("expected_source_substrings") or [])
    expected_titles = len(case.get("expected_title_substrings") or [])
    expected_answer = len(case.get("expected_answer_substrings") or [])
    forbidden = len(case.get("forbidden_substrings") or [])
    return {
        "id": case["id"],
        "query": case["query"],
        "passed": False,
        "failures": [f"request/evaluation error: {error}"],
        "route": None,
        "sources": [],
        "latency_ms": round(latency_ms, 2) if latency_ms is not None else None,
        "server_total_ms": None,
        "answer_preview": "",
        "rag_metrics": {},
        "checks": {
            "route": False if case.get("expected_route") else None,
            "sources": {"matched": 0, "expected": expected_sources},
            "titles": {"matched": 0, "expected": expected_titles},
            "answer_substrings": {"matched": 0, "expected": expected_answer},
            "forbidden_substrings": {"found": 0, "expected_absent": forbidden},
            "latency": False if case.get("max_latency_ms") is not None else None,
        },
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    route_checks = [
        result["checks"]["route"]
        for result in results
        if result["checks"]["route"] is not None
    ]
    source_matched = sum(result["checks"]["sources"]["matched"] for result in results)
    source_expected = sum(result["checks"]["sources"]["expected"] for result in results)
    title_matched = sum(result["checks"]["titles"]["matched"] for result in results)
    title_expected = sum(result["checks"]["titles"]["expected"] for result in results)
    latencies = [
        float(result["latency_ms"])
        for result in results
        if isinstance(result.get("latency_ms"), int | float)
        and not isinstance(result.get("latency_ms"), bool)
    ]
    route_correct = sum(value is True for value in route_checks)
    passed = sum(result["passed"] is True for result in results)
    return {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "route_accuracy": {
            "correct": route_correct,
            "expected": len(route_checks),
            "value": _ratio(route_correct, len(route_checks)),
        },
        "source_recall": {
            "matched": source_matched,
            "expected": source_expected,
            "value": _ratio(source_matched, source_expected),
        },
        "title_recall": {
            "matched": title_matched,
            "expected": title_expected,
            "value": _ratio(title_matched, title_expected),
        },
        "latency_ms": _latency_summary(latencies),
    }


def validate_config(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a JSON object")
    if "api_key" in raw:
        raise ValueError("do not store api_key in the config; use api_key_env")
    cases = raw.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("configuration must contain a non-empty cases list")

    normalized = dict(raw)
    normalized_cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, value in enumerate(cases, start=1):
        if not isinstance(value, dict):
            raise ValueError(f"case {index} must be a JSON object")
        case = dict(value)
        query = case.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"case {index} has no non-empty query")
        case["query"] = query.strip()
        case_id = str(case.get("id") or f"case-{index}").strip()
        if not case_id or case_id in seen_ids:
            raise ValueError(f"case {index} has an empty or duplicate id: {case_id!r}")
        seen_ids.add(case_id)
        case["id"] = case_id

        history = case.get("history") or []
        if not isinstance(history, list):
            raise ValueError(f"case {case_id!r} history must be a list")
        for message_index, message in enumerate(history, start=1):
            if not isinstance(message, dict):
                raise ValueError(f"case {case_id!r} history item {message_index} must be an object")
            if message.get("role") not in {"system", "user", "assistant"}:
                raise ValueError(f"case {case_id!r} history item {message_index} has an invalid role")
            if not isinstance(message.get("content"), str):
                raise ValueError(f"case {case_id!r} history item {message_index} content must be text")

        expected_route = case.get("expected_route")
        if expected_route is not None and (not isinstance(expected_route, str) or not expected_route.strip()):
            raise ValueError(f"case {case_id!r} expected_route must be non-empty text")
        if isinstance(expected_route, str):
            case["expected_route"] = expected_route.strip()

        for field in EXPECTED_LIST_FIELDS:
            values = case.get(field, [])
            if values is None:
                values = []
            if not isinstance(values, list) or not all(
                isinstance(item, str) and item.strip() for item in values
            ):
                raise ValueError(f"case {case_id!r} {field} must be a list of non-empty strings")
            case[field] = [item.strip() for item in values]

        max_latency = case.get("max_latency_ms")
        if max_latency is not None and (
            not isinstance(max_latency, int | float)
            or isinstance(max_latency, bool)
            or max_latency <= 0
        ):
            raise ValueError(f"case {case_id!r} max_latency_ms must be a positive number")
        normalized_cases.append(case)

    timeout = normalized.get("timeout_seconds", 180)
    if (
        not isinstance(timeout, int | float)
        or isinstance(timeout, bool)
        or timeout <= 0
    ):
        raise ValueError("timeout_seconds must be a positive number")
    normalized["timeout_seconds"] = float(timeout)
    normalized["cases"] = normalized_cases
    return normalized


def call_chat_api(
    *,
    endpoint: str,
    model: str,
    messages: list[dict[str, str]],
    api_key: str,
    timeout_seconds: float,
) -> ApiObservation:
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "stream": False,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "glavstroy-rag-golden-eval/1.0",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(endpoint, data=body, headers=headers, method="POST")
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw_body = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"HTTP {exc.code}: {detail or exc.reason}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"could not call {endpoint}: {exc}") from exc
    latency_ms = (time.perf_counter() - started) * 1000
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("API returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("API returned a non-object JSON response")
    return parse_api_response(payload, latency_ms=latency_ms)


def run_evaluation(
    config: dict[str, Any],
    *,
    endpoint: str,
    model: str,
    api_key: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for case in config["cases"]:
        started = time.perf_counter()
        try:
            observation = call_chat_api(
                endpoint=endpoint,
                model=model,
                messages=build_messages(case),
                api_key=api_key,
                timeout_seconds=timeout_seconds,
            )
            result = evaluate_case(case, observation)
        except Exception as exc:
            result = evaluation_error_result(
                case,
                exc,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        results.append(result)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "model": model,
        "summary": summarize_results(results),
        "cases": results,
    }


def _chat_endpoint(base_url: str) -> str:
    parts = urlsplit(base_url.strip())
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError("api_base_url must be an absolute http(s) URL")
    path = parts.path.rstrip("/")
    if path.endswith("/chat/completions"):
        final_path = path
    elif path.endswith("/v1"):
        final_path = f"{path}/chat/completions"
    else:
        final_path = f"{path}/v1/chat/completions"
    return urlunsplit((parts.scheme, parts.netloc, final_path, parts.query, ""))


def _contains(needle: str, haystack: str, case_sensitive: bool) -> bool:
    if case_sensitive:
        return needle in haystack
    return needle.casefold() in haystack.casefold()


def _contains_in_any(needle: str, values: list[str], case_sensitive: bool) -> bool:
    return any(_contains(needle, value, case_sensitive) for value in values)


def _equal_text(expected: str, actual: str, case_sensitive: bool) -> bool:
    if case_sensitive:
        return expected == actual
    return expected.casefold() == actual.casefold()


def _preview(text: str, *, limit: int = 500) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def _latency_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": round(statistics.fmean(ordered), 2),
        "p50": round(statistics.median(ordered), 2),
        "p95": round(_nearest_rank_percentile(ordered, 95), 2),
        "max": round(max(ordered), 2),
    }


def _nearest_rank_percentile(ordered: list[float], percentile: int) -> float:
    rank = max(1, math.ceil((percentile / 100) * len(ordered)))
    return ordered[rank - 1]


def _print_human_report(report: dict[str, Any]) -> None:
    for result in report["cases"]:
        status = "PASS" if result["passed"] else "FAIL"
        route = result.get("route") or "-"
        latency = result.get("latency_ms")
        latency_text = f"{latency:.2f} ms" if isinstance(latency, int | float) else "n/a"
        print(f"{status} {result['id']} route={route} latency={latency_text}")
        for failure in result["failures"]:
            print(f"  - {failure}")

    summary = report["summary"]
    route = summary["route_accuracy"]
    source = summary["source_recall"]
    latency = summary["latency_ms"]
    print(
        "Summary: "
        f"passed={summary['passed']}/{summary['total']} "
        f"route_accuracy={_format_metric(route['value'])} "
        f"source_recall={_format_metric(source['value'])} "
        f"latency_p50={_format_ms(latency['p50'])} "
        f"latency_p95={_format_ms(latency['p95'])}"
    )


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def _format_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}ms"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a golden evaluation set against the OpenAI-compatible RAG API."
    )
    parser.add_argument("--config", required=True, help="Path to a golden evaluation JSON file.")
    parser.add_argument(
        "--base-url",
        help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8000/v1.",
    )
    parser.add_argument("--model", help="Model id sent in chat completion requests.")
    parser.add_argument(
        "--api-key-env",
        help="Name of the environment variable containing the API key.",
    )
    parser.add_argument("--timeout-seconds", type=float, help="Per-request HTTP timeout.")
    parser.add_argument("--report-json", help="Optional path for the complete JSON report.")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        raw_config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        config = validate_config(raw_config)
        base_url = (
            args.base_url
            or os.getenv("RAG_EVAL_API_BASE_URL")
            or config.get("api_base_url")
            or "http://127.0.0.1:8000/v1"
        )
        endpoint = _chat_endpoint(str(base_url))
        model = str(
            args.model
            or os.getenv("RAG_EVAL_MODEL")
            or config.get("model")
            or "document-search-rag"
        )
        api_key_env = str(args.api_key_env or config.get("api_key_env") or "OPENAI_COMPAT_API_KEY")
        api_key = os.getenv(api_key_env, "")
        timeout_seconds = float(
            args.timeout_seconds
            if args.timeout_seconds is not None
            else config["timeout_seconds"]
        )
        if timeout_seconds <= 0:
            raise ValueError("timeout must be positive")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    report = run_evaluation(
        config,
        endpoint=endpoint,
        model=model,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )
    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_human_report(report)
    return 1 if report["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
