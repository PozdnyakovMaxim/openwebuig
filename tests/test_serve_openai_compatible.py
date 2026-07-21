from __future__ import annotations

from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from serve_openai_compatible import validate_bind_auth


class ServeOpenAICompatibleTest(unittest.TestCase):
    def test_external_bind_requires_a_real_shared_secret(self) -> None:
        for key in ("", "short", "replace_with_a_strong_shared_secret"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "at least 16"):
                validate_bind_auth("0.0.0.0", key)

    def test_external_bind_accepts_long_secret(self) -> None:
        validate_bind_auth("0.0.0.0", "a-real-shared-secret-value")

    def test_loopback_bind_can_be_used_without_auth_for_local_development(self) -> None:
        validate_bind_auth("127.0.0.1", "")
        validate_bind_auth("::1", "")
        validate_bind_auth("localhost", "")


if __name__ == "__main__":
    unittest.main()
