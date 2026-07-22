from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def find_latest_report(root: Path = ROOT) -> Path:
    reports = list((root / "artifacts").glob("candidate-*/audit-before.json"))
    if not reports:
        raise FileNotFoundError(
            "Не найден artifacts/candidate-*/audit-before.json. Сначала запустите аудит кандидата."
        )
    return max(reports, key=lambda path: path.stat().st_mtime)


def format_first_failure(report: dict[str, Any]) -> str:
    issues = list(report.get("issues") or [])
    matching = [
        issue
        for issue in issues
        if issue.get("code") == "source_ooxml_text_missing"
        and issue.get("level") == "error"
    ]
    if not matching:
        matching = [
            issue
            for issue in issues
            if issue.get("code") == "source_ooxml_text_missing"
        ]
    if not matching:
        return "Ошибок source_ooxml_text_missing в отчёте нет."

    issue = matching[0]
    details = dict(issue.get("details") or {})
    lines = [
        f"Уровень: {issue.get('level') or 'unknown'}",
        f"Документ: {issue.get('source_name') or issue.get('doc_id') or 'unknown'}",
        f"Причина: {issue.get('message') or 'unknown'}",
    ]
    samples = list(details.get("segment_samples") or [])
    if samples:
        lines.append("Пропущенные фрагменты:")
        for index, sample in enumerate(samples, start=1):
            lines.append(
                f"{index}. [{sample.get('story') or 'unknown'}/"
                f"{sample.get('location') or 'unknown'}] {sample.get('text') or ''}"
            )
    else:
        token_counts = dict(details.get("missing_token_counts") or {})
        lines.append(f"Пропущенные токены: {json.dumps(token_counts, ensure_ascii=False)}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Показать первый критический пропуск из последнего аудита корпуса."
    )
    parser.add_argument("report", nargs="?", help="Необязательный путь к audit-before.json")
    args = parser.parse_args()

    report_path = Path(args.report).expanduser().resolve() if args.report else find_latest_report()
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Не удалось прочитать {report_path}: {exc}") from exc

    print(f"Отчёт: {report_path}")
    print(format_first_failure(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
