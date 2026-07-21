from __future__ import annotations

import importlib.util
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "apply_openwebui_branding.py"


def load_branding_module():
    spec = importlib.util.spec_from_file_location("apply_openwebui_branding", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OpenWebUIBrandingTests(unittest.TestCase):
    def run_patcher(
        self,
        db_path: Path,
        *,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        patcher_path = db_path.parent / "patch_branding.py"
        config_path = db_path.parent / "config.json"
        load_branding_module().write_patcher(patcher_path)
        config_path.write_text(
            json.dumps(
                {
                    "brand_name": "ГлавстройLLM",
                    "technical_label": "project-ui",
                    "model_id": "document-search-rag",
                    "default_model_id": "document-search-rag",
                    "db_path": str(db_path),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return subprocess.run(
            [sys.executable, str(patcher_path), str(config_path)],
            check=check,
            capture_output=True,
            text=True,
        )

    def test_patcher_supports_current_per_key_config_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "webui.db"

            with closing(sqlite3.connect(db_path)) as con, con:
                con.execute(
                    "create table config (key text primary key, value text not null, updated_at integer)"
                )
                con.execute(
                    "insert into config (key, value, updated_at) values (?, ?, ?)",
                    ("ui.default_models", json.dumps("old-model"), 0),
                )
                con.execute(
                    "insert into config (key, value, updated_at) values (?, ?, ?)",
                    ("openai.api_keys", json.dumps(["keep-this-key"]), 0),
                )
                con.execute(
                    "create table user (id text primary key, settings text, created_at integer, updated_at integer)"
                )
                con.execute(
                    "insert into user (id, settings, created_at, updated_at) values (?, ?, ?, ?)",
                    ("user-1", "{}", 0, 0),
                )
                con.execute(
                    """
                    create table model (
                        id text primary key,
                        user_id text,
                        base_model_id text,
                        name text,
                        params text,
                        meta text,
                        access_control text,
                        is_active integer,
                        created_at integer,
                        updated_at integer
                    )
                    """
                )
                con.execute(
                    "insert into model (id, user_id, name, params, meta, is_active, created_at, updated_at) values (?, ?, ?, ?, ?, ?, ?, ?)",
                    ("document-search-rag", "user-1", "Old", "{}", "{}", 1, 0, 0),
                )

            result = self.run_patcher(db_path)

            with closing(sqlite3.connect(db_path)) as con:
                persistent = {
                    key: json.loads(value)
                    for key, value in con.execute("select key, value from config").fetchall()
                }
                user_settings = json.loads(
                    con.execute("select settings from user where id = 'user-1'").fetchone()[0]
                )
                model = con.execute(
                    "select name, meta from model where id = 'document-search-rag'"
                ).fetchone()

            self.assertEqual(persistent["ui.default_models"], "document-search-rag")
            self.assertEqual(
                persistent["ui.default_pinned_models"], "document-search-rag"
            )
            self.assertEqual(persistent["ui.model_order_list"], ["document-search-rag"])
            self.assertFalse(persistent["evaluation.arena.enable"])
            self.assertEqual(persistent["openai.api_keys"], ["keep-this-key"])
            self.assertEqual(user_settings["ui"]["default_model"], "document-search-rag")
            self.assertEqual(model[0], "ГлавстройLLM")
            self.assertEqual(json.loads(model[1])["profile_image_url"], "/static/brand-icon.svg")
            self.assertIn("patched_runtime_db=true", result.stdout)

    def test_patcher_supports_legacy_config_without_touching_connection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "webui.db"
            original = {
                "openai": {"api_keys": ["keep-this-key"]},
                "ldap": {"enable": True},
            }
            with closing(sqlite3.connect(db_path)) as con, con:
                con.execute(
                    "create table config (id integer primary key, data text, updated_at datetime)"
                )
                con.execute(
                    "insert into config (id, data, updated_at) values (?, ?, ?)",
                    (1, json.dumps(original), 0),
                )

            result = self.run_patcher(db_path)

            with closing(sqlite3.connect(db_path)) as con:
                raw, updated_at = con.execute(
                    "select data, updated_at from config"
                ).fetchone()
                repaired = json.loads(raw)
            self.assertEqual(repaired["openai"]["api_keys"], ["keep-this-key"])
            self.assertTrue(repaired["ldap"]["enable"])
            self.assertEqual(repaired["ui"]["default_model"], "document-search-rag")
            self.assertIsInstance(updated_at, str)
            self.assertIn("patched_runtime_db=true", result.stdout)

    def test_patcher_rejects_malformed_legacy_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "webui.db"
            with closing(sqlite3.connect(db_path)) as con, con:
                con.execute(
                    "create table config (id integer primary key, data text, updated_at integer)"
                )
                con.execute(
                    "insert into config (id, data, updated_at) values (?, ?, ?)",
                    (1, "{malformed", 0),
                )

            result = self.run_patcher(db_path, check=False)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to overwrite", result.stderr)
            with closing(sqlite3.connect(db_path)) as con:
                self.assertEqual(
                    con.execute("select data from config").fetchone()[0],
                    "{malformed",
                )


if __name__ == "__main__":
    unittest.main()
