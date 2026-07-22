from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "fix_openwebui_model_access.py"


def load_inner_script() -> str:
    spec = importlib.util.spec_from_file_location("fix_openwebui_model_access", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.INNER_SCRIPT


class OpenWebUIAccessRepairTests(unittest.TestCase):
    def run_inner(
        self,
        *,
        api_key: str | None,
        config_schema: str = "legacy",
        legacy_raw: str | None = None,
        legacy_row: bool = True,
        verify_upstream: bool = False,
        api_base_url: str = "http://host.docker.internal:8000/v1/",
        loader_api_key: str | None = None,
    ) -> tuple[dict, str, Path, int]:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        root = Path(temp_dir.name)
        db_path = root / "webui.db"
        inner_path = root / "fix_inner.py"
        config_path = root / "config.json"

        old_config = {
            "ldap": {"enable": True},
            "openai": {
                "enable": True,
                "api_base_urls": ["http://host.docker.internal:8000/v1"],
                "api_keys": ["anything"],
                "api_configs": {"0": {"enable": False, "prefix_id": ""}},
            }
        }
        with closing(sqlite3.connect(db_path)) as con, con:
            if config_schema == "legacy":
                con.execute(
                    """
                    create table config (
                        id integer primary key,
                        data text,
                        version integer,
                        created_at datetime,
                        updated_at datetime
                    )
                    """
                )
                if legacy_row:
                    con.execute(
                        "insert into config (id, data, updated_at) values (?, ?, ?)",
                        (1, legacy_raw if legacy_raw is not None else json.dumps(old_config), 0),
                    )
            elif config_schema in {"per-key", "per-key-no-updated-at"}:
                updated_at_column = ", updated_at integer" if config_schema == "per-key" else ""
                con.execute(
                    f"create table config (key text primary key, value text not null{updated_at_column})"
                )
                for key, value in {
                    "ldap.enable": True,
                    "openai.enable": True,
                    "openai.api_base_urls": ["http://host.docker.internal:8000/v1"],
                    "openai.api_keys": ["anything"],
                    "openai.api_configs": {"0": {"enable": False, "prefix_id": ""}},
                }.items():
                    if config_schema == "per-key":
                        con.execute(
                            "insert into config (key, value, updated_at) values (?, ?, ?)",
                            (key, json.dumps(value), 0),
                        )
                    else:
                        con.execute(
                            "insert into config (key, value) values (?, ?)",
                            (key, json.dumps(value)),
                        )
            else:
                raise ValueError(config_schema)

        inner_path.write_text(load_inner_script(), encoding="utf-8")
        config_path.write_text(
            json.dumps(
                {
                    "db_path": str(db_path),
                    "model_id": "document-search-rag",
                    "custom_model_id": "glavstroy-llm",
                    "model_name": "ГлавстройLLM",
                    "activate_pending": False,
                    "verify_upstream": verify_upstream,
                }
            ),
            encoding="utf-8",
        )
        env = os.environ.copy()
        env.pop("OPENAI_COMPAT_API_KEY", None)
        env.pop("OPENAI_COMPAT_API_BASE_URL", None)
        env.pop("OPENAI_API_KEY", None)
        env.pop("OPENAI_API_KEYS", None)
        env.pop("EXTERNAL_DOCUMENT_LOADER_API_KEY", None)
        env["OPENAI_API_BASE_URL"] = api_base_url
        if api_key is not None:
            env["OPENAI_API_KEY"] = api_key
        if loader_api_key is not None:
            env["EXTERNAL_DOCUMENT_LOADER_API_KEY"] = loader_api_key

        result = subprocess.run(
            [sys.executable, str(inner_path), str(config_path)],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        with closing(sqlite3.connect(db_path)) as con:
            if config_schema == "legacy":
                raw = con.execute("select data from config").fetchone()[0]
                try:
                    repaired = json.loads(raw)
                except json.JSONDecodeError:
                    repaired = {"__raw__": raw}
            else:
                repaired = {
                    key: json.loads(value)
                    for key, value in con.execute("select key, value from config").fetchall()
                }
        return repaired, result.stdout + result.stderr, db_path, result.returncode

    def start_models_server(self, expected_key: str) -> tuple[ThreadingHTTPServer, str]:
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path != "/v1/models":
                    self.send_error(404)
                    return
                if self.headers.get("Authorization") != f"Bearer {expected_key}":
                    self.send_error(401)
                    return
                payload = json.dumps({"data": [{"id": "document-search-rag"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server, f"http://127.0.0.1:{server.server_port}/v1"

    def test_updates_persistent_openai_connection_from_container_environment(self) -> None:
        repaired, output, db_path, returncode = self.run_inner(
            api_key="new-secret-at-least-16"
        )

        self.assertEqual(returncode, 0, output)
        self.assertEqual(
            repaired["openai"]["api_base_urls"],
            ["http://host.docker.internal:8000/v1"],
        )
        self.assertEqual(repaired["openai"]["api_keys"], ["new-secret-at-least-16"])
        self.assertTrue(repaired["openai"]["enable"])
        self.assertEqual(
            repaired["openai"]["api_configs"],
            {},
        )
        self.assertEqual(repaired["rag"]["content_extraction_engine"], "external")
        self.assertEqual(repaired["rag"]["CONTENT_EXTRACTION_ENGINE"], "external")
        self.assertEqual(
            repaired["rag"]["external_document_loader_url"],
            "http://host.docker.internal:8000",
        )
        self.assertEqual(
            repaired["rag"]["external_document_loader_api_key"],
            "new-secret-at-least-16",
        )
        self.assertTrue(repaired["ldap"]["enable"])
        self.assertIn("openai_connection_updated=1", output)
        self.assertIn("external_document_loader_configured=1", output)
        self.assertIn("openai_connection_verified=1", output)
        self.assertNotIn("new-secret-at-least-16", output)
        self.assertEqual(len(list(db_path.parent.glob("webui.db.backup-*"))), 1)

    def test_current_schema_without_updated_at_is_supported(self) -> None:
        repaired, output, _, returncode = self.run_inner(
            api_key="new-secret-at-least-16",
            config_schema="per-key-no-updated-at",
        )

        self.assertEqual(returncode, 0, output)
        self.assertEqual(repaired["openai.api_keys"], ["new-secret-at-least-16"])

    def test_dedicated_document_loader_key_is_persisted_separately(self) -> None:
        repaired, output, _, returncode = self.run_inner(
            api_key="compat-secret-at-least-16",
            config_schema="per-key",
            loader_api_key="loader-secret-at-least-16",
        )

        self.assertEqual(returncode, 0, output)
        self.assertEqual(repaired["openai.api_keys"], ["compat-secret-at-least-16"])
        self.assertEqual(
            repaired["rag.external_document_loader_api_key"],
            "loader-secret-at-least-16",
        )
        self.assertNotIn("loader-secret-at-least-16", output)

    def test_repair_shell_reads_and_forwards_dedicated_loader_key(self) -> None:
        repair_script = (ROOT / "scripts" / "repair_openwebui_access.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn('loader_api_key="${OPENWEBUI_DOCUMENT_LOADER_API_KEY:-}"', repair_script)
        self.assertIn('EXTERNAL_DOCUMENT_LOADER_API_KEY="$loader_api_key"', repair_script)

    def test_inserts_missing_legacy_config_row(self) -> None:
        repaired, output, db_path, returncode = self.run_inner(
            api_key="new-secret-at-least-16",
            legacy_row=False,
        )

        self.assertEqual(returncode, 0, output)
        self.assertEqual(
            repaired["openai"]["api_base_urls"],
            ["http://host.docker.internal:8000/v1"],
        )
        self.assertEqual(repaired["default_model"], "document-search-rag")
        with closing(sqlite3.connect(db_path)) as con:
            created_at, updated_at = con.execute(
                "select created_at, updated_at from config"
            ).fetchone()
        self.assertIsInstance(created_at, str)
        self.assertIsInstance(updated_at, str)

    def test_rejects_malformed_legacy_config_without_overwriting_it(self) -> None:
        repaired, output, db_path, returncode = self.run_inner(
            api_key="new-secret-at-least-16",
            legacy_raw="{malformed",
        )

        self.assertNotEqual(returncode, 0)
        self.assertEqual(repaired["__raw__"], "{malformed")
        self.assertIn("refusing to overwrite", output)
        self.assertEqual(len(list(db_path.parent.glob("webui.db.backup-*"))), 1)

    def test_updates_current_per_key_config_schema(self) -> None:
        repaired, output, _, returncode = self.run_inner(
            api_key="new-secret-at-least-16",
            config_schema="per-key",
        )

        self.assertEqual(returncode, 0, output)
        self.assertEqual(
            repaired["openai.api_base_urls"],
            ["http://host.docker.internal:8000/v1"],
        )
        self.assertEqual(repaired["openai.api_keys"], ["new-secret-at-least-16"])
        self.assertTrue(repaired["openai.enable"])
        self.assertEqual(
            repaired["openai.api_configs"],
            {},
        )
        self.assertEqual(repaired["rag.content_extraction_engine"], "external")
        self.assertEqual(
            repaired["rag.external_document_loader_url"],
            "http://host.docker.internal:8000",
        )
        self.assertEqual(
            repaired["rag.external_document_loader_api_key"],
            "new-secret-at-least-16",
        )
        self.assertTrue(repaired["ldap.enable"])
        self.assertEqual(repaired["ui.default_models"], "document-search-rag")
        self.assertEqual(repaired["ui.model_order_list"], ["document-search-rag"])
        self.assertIn("openai_connection_updated=1", output)

    def test_leaves_persistent_connection_unchanged_without_key(self) -> None:
        repaired, output, _, returncode = self.run_inner(api_key=None)

        self.assertNotEqual(returncode, 0)
        self.assertEqual(repaired["openai"]["api_keys"], ["anything"])
        self.assertFalse(repaired["openai"]["api_configs"]["0"]["enable"])
        self.assertIn("API key is missing", output)
        self.assertNotIn("backup=", output)

    def test_upstream_preflight_uses_key_and_accepts_expected_model(self) -> None:
        expected_key = "right-key-at-least-16"
        _, base_url = self.start_models_server(expected_key)

        repaired, output, _, returncode = self.run_inner(
            api_key=expected_key,
            config_schema="per-key",
            verify_upstream=True,
            api_base_url=base_url,
        )

        self.assertEqual(returncode, 0, output)
        self.assertEqual(repaired["openai.api_keys"], [expected_key])
        self.assertIn("upstream_model_verified=document-search-rag", output)
        self.assertNotIn(expected_key, output)

    def test_failed_upstream_preflight_does_not_change_database(self) -> None:
        _, base_url = self.start_models_server("right-key-at-least-16")

        repaired, output, db_path, returncode = self.run_inner(
            api_key="wrong-key-at-least-16",
            config_schema="per-key",
            verify_upstream=True,
            api_base_url=base_url,
        )

        self.assertNotEqual(returncode, 0)
        self.assertEqual(repaired["openai.api_keys"], ["anything"])
        self.assertIn("HTTP Error 401", output)
        self.assertNotIn("wrong-key-at-least-16", output)
        self.assertEqual(list(db_path.parent.glob("webui.db.backup-*")), [])


if __name__ == "__main__":
    unittest.main()
