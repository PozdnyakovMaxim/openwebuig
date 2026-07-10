#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import tempfile
from pathlib import Path


INNER_SCRIPT = r'''
from __future__ import annotations

import json
import shutil
import sqlite3
import sys
import time
import uuid
from pathlib import Path


config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
db_path = config["db_path"]
model_id = config["model_id"]
custom_model_id = config["custom_model_id"]
model_name = config["model_name"]
activate_pending = config["activate_pending"]

db_file = Path(db_path)
if not db_file.exists():
    raise SystemExit(f"Open WebUI database was not found: {db_path}")

backup_path = db_file.with_name(f"{db_file.name}.backup-{int(time.time())}")
shutil.copy2(db_file, backup_path)
print(f"backup={backup_path}")

con = sqlite3.connect(db_path)
con.row_factory = sqlite3.Row


def tables() -> set[str]:
    rows = con.execute("select name from sqlite_master where type='table'").fetchall()
    return {row["name"] for row in rows}


def columns(table: str) -> set[str]:
    rows = con.execute(f"pragma table_info({table})").fetchall()
    return {row["name"] for row in rows}


def safe_json(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


existing_tables = tables()
changed_config = 0
changed_users = 0
activated_users = 0
changed_models = 0
changed_grants = 0

if "config" in existing_tables:
    config_columns = columns("config")
    if {"id", "data"}.issubset(config_columns):
        row = con.execute("select id, data from config order by id limit 1").fetchone()
        now = int(time.time())
        if row:
            data = safe_json(row["data"])
            data["default_model"] = custom_model_id
            data["default_models"] = [custom_model_id]
            data["model_order_list"] = [custom_model_id]
            data["title"] = model_name
            data["name"] = model_name
            data["default_prompt_suggestions"] = []
            ui = data.setdefault("ui", {})
            ui["default_model"] = custom_model_id
            ui["default_models"] = [custom_model_id]
            ui["model_order_list"] = [custom_model_id]
            ui["prompt_suggestions"] = []
            query = "update config set data = ?"
            params = [json.dumps(data, ensure_ascii=False)]
            if "updated_at" in config_columns:
                query += ", updated_at = ?"
                params.append(now)
            query += " where id = ?"
            params.append(row["id"])
            con.execute(query, params)
            changed_config = 1

if "model" in existing_tables:
    model_columns = columns("model")
    now = int(time.time())
    meta = {
        "profile_image_url": "/static/brand-icon.svg",
        "description": "Поиск и ответы по документам",
        "capabilities": {"vision": False, "citations": True},
    }
    user_id = "system"
    if "user" in existing_tables:
        user_row = con.execute("select id from user order by created_at limit 1").fetchone()
        if user_row:
            user_id = user_row["id"]

    def upsert_model(row_id: str, *, base_model_id: str | None, name: str) -> None:
        global changed_models
        existing = con.execute("select id from model where id = ?", (row_id,)).fetchone()
        values = {
            "user_id": user_id,
            "base_model_id": base_model_id,
            "name": name,
            "meta": json.dumps(meta, ensure_ascii=False),
            "params": "{}",
            "access_control": None,
            "is_active": 1,
            "updated_at": now,
        }
        if existing:
            assignments = []
            params = []
            for column, value in values.items():
                if column in model_columns:
                    assignments.append(f"{column} = ?")
                    params.append(value)
            if assignments:
                params.append(row_id)
                con.execute(f"update model set {', '.join(assignments)} where id = ?", params)
                changed_models += 1
            return

        values["id"] = row_id
        values["created_at"] = now
        insert_columns = [column for column in values if column in model_columns]
        placeholders = ", ".join("?" for _ in insert_columns)
        con.execute(
            f"insert into model ({', '.join(insert_columns)}) values ({placeholders})",
            [values[column] for column in insert_columns],
        )
        changed_models += 1

    upsert_model(model_id, base_model_id=None, name=model_id)
    upsert_model(custom_model_id, base_model_id=model_id, name=model_name)

if "access_grant" in existing_tables:
    grant_columns = columns("access_grant")

    def ensure_public_read(resource_id: str) -> None:
        global changed_grants
        required = {
            "resource_type",
            "resource_id",
            "principal_type",
            "principal_id",
            "permission",
        }
        if not required.issubset(grant_columns):
            return
        existing = con.execute(
            """
            select id from access_grant
            where resource_type = ?
              and resource_id = ?
              and principal_type = ?
              and principal_id = ?
              and permission = ?
            """,
            ("model", resource_id, "user", "*", "read"),
        ).fetchone()
        if existing:
            return
        values = {
            "id": str(uuid.uuid4()),
            "resource_type": "model",
            "resource_id": resource_id,
            "principal_type": "user",
            "principal_id": "*",
            "permission": "read",
            "created_at": int(time.time()),
        }
        insert_columns = [column for column in values if column in grant_columns]
        placeholders = ", ".join("?" for _ in insert_columns)
        con.execute(
            f"insert into access_grant ({', '.join(insert_columns)}) values ({placeholders})",
            [values[column] for column in insert_columns],
        )
        changed_grants += 1

    ensure_public_read(model_id)
    ensure_public_read(custom_model_id)

if "user" in existing_tables:
    user_columns = columns("user")
    if {"id", "settings"}.issubset(user_columns):
        rows = con.execute("select id, settings, role from user" if "role" in user_columns else "select id, settings from user").fetchall()
        for row in rows:
            settings = safe_json(row["settings"])
            settings["default_model"] = custom_model_id
            settings["default_models"] = [custom_model_id]
            settings["model_order_list"] = [custom_model_id]
            ui = settings.setdefault("ui", {})
            ui["default_model"] = custom_model_id
            ui["default_models"] = [custom_model_id]
            ui["model_order_list"] = [custom_model_id]
            query = "update user set settings = ?"
            params = [json.dumps(settings, ensure_ascii=False)]
            if "updated_at" in user_columns:
                query += ", updated_at = ?"
                params.append(int(time.time()))
            if activate_pending and "role" in user_columns and row["role"] == "pending":
                query += ", role = ?"
                params.append("user")
                activated_users += 1
            query += " where id = ?"
            params.append(row["id"])
            con.execute(query, params)
            changed_users += 1

con.commit()
con.close()

print(f"config_updated={changed_config}")
print(f"models_updated={changed_models}")
print(f"public_grants_added={changed_grants}")
print(f"users_updated={changed_users}")
print(f"pending_activated={activated_users}")
'''


def run(command: list[str]) -> None:
    print("+", shlex.join(command))
    subprocess.run(command, check=True, text=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fix model visibility and defaults in Open WebUI.")
    parser.add_argument("--container", default="document-search-webui")
    parser.add_argument("--model-id", default="document-search-rag")
    parser.add_argument("--custom-model-id", default="glavstroy-llm")
    parser.add_argument("--model-name", default="ГлавстройLLM")
    parser.add_argument("--db-path", default="/app/backend/data/webui.db")
    parser.add_argument("--activate-pending", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        inner = tmp_dir / "fix_openwebui_model_access_inner.py"
        config = tmp_dir / "fix_openwebui_model_access.json"
        inner.write_text(INNER_SCRIPT, encoding="utf-8")
        config.write_text(
            json.dumps(
                {
                    "db_path": args.db_path,
                    "model_id": args.model_id,
                    "custom_model_id": args.custom_model_id,
                    "model_name": args.model_name,
                    "activate_pending": args.activate_pending,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        run(["docker", "cp", str(inner), f"{args.container}:/tmp/{inner.name}"])
        run(["docker", "cp", str(config), f"{args.container}:/tmp/{config.name}"])
        run(["docker", "exec", args.container, "python", f"/tmp/{inner.name}", f"/tmp/{config.name}"])


if __name__ == "__main__":
    main()
