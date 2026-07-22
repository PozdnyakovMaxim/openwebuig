from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "show_audit_failure.py"
SPEC = importlib.util.spec_from_file_location("show_audit_failure", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ShowAuditFailureTest(unittest.TestCase):
    def test_prefers_error_and_renders_nested_samples(self) -> None:
        report = {
            "issues": [
                {
                    "level": "warning",
                    "code": "source_ooxml_text_missing",
                    "source_name": "warning.docx",
                },
                {
                    "level": "error",
                    "code": "source_ooxml_text_missing",
                    "source_name": "policy.docx",
                    "message": "visible source text is missing",
                    "details": {
                        "segment_samples": [
                            {
                                "story": "body",
                                "location": "textbox",
                                "text": "Критический фрагмент",
                            }
                        ]
                    },
                },
            ]
        }

        output = MODULE.format_first_failure(report)

        self.assertIn("Уровень: error", output)
        self.assertIn("Документ: policy.docx", output)
        self.assertIn("[body/textbox] Критический фрагмент", output)
        self.assertNotIn("warning.docx", output)

    def test_finds_newest_report(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            older = root / "artifacts" / "candidate-old" / "audit-before.json"
            newer = root / "artifacts" / "candidate-new" / "audit-before.json"
            older.parent.mkdir(parents=True)
            newer.parent.mkdir(parents=True)
            older.write_text("{}", encoding="utf-8")
            newer.write_text("{}", encoding="utf-8")
            older.touch()
            newer.touch()
            older_stat = older.stat()
            newer_stat = newer.stat()
            older.touch()
            newer.touch()
            # Explicit mtimes keep the test deterministic on coarse filesystems.
            import os

            os.utime(older, (older_stat.st_atime, 1))
            os.utime(newer, (newer_stat.st_atime, 2))

            self.assertEqual(MODULE.find_latest_report(root), newer)


if __name__ == "__main__":
    unittest.main()
