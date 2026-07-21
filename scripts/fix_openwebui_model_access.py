#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path


INNER_SCRIPT = r'''
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
import urllib.request
import uuid
from pathlib import Path


config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
db_path = config["db_path"]
model_id = config["model_id"]
custom_model_id = config["custom_model_id"]
model_name = config["model_name"]
activate_pending = config["activate_pending"]
verify_upstream = config.get("verify_upstream", True)


def first_env_value(single_name: str, plural_name: str) -> str:
    single_value = os.environ.get(single_name, "").strip()
    if single_value:
        return single_value
    return next(
        (
            value.strip()
            for value in os.environ.get(plural_name, "").split(";")
            if value.strip()
        ),
        "",
    )


openai_api_key = (
    os.environ.get("OPENAI_COMPAT_API_KEY", "").strip()
    or first_env_value("OPENAI_API_KEY", "OPENAI_API_KEYS")
)
openai_api_base_url = (
    os.environ.get("OPENAI_COMPAT_API_BASE_URL", "").strip()
    or first_env_value("OPENAI_API_BASE_URL", "OPENAI_API_BASE_URLS")
    or "http://host.docker.internal:8000/v1"
).rstrip("/")

if not openai_api_key:
    raise SystemExit("OpenAI-compatible API key is missing; persistent config was not changed")

if verify_upstream:
    request = urllib.request.Request(
        f"{openai_api_base_url}/models",
        headers={"Authorization": f"Bearer {openai_api_key}"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        models_payload = json.loads(response.read().decode("utf-8"))
    model_ids = {
        item.get("id")
        for item in models_payload.get("data", [])
        if isinstance(item, dict)
    }
    if model_id not in model_ids:
        raise SystemExit(f"Upstream model is missing: {model_id}")

db_file = Path(db_path)
if not db_file.exists():
    raise SystemExit(f"Open WebUI database was not found: {db_path}")

backup_path = db_file.with_name(f"{db_file.name}.backup-{time.time_ns()}")
backup_source = sqlite3.connect(db_path)
backup_target = sqlite3.connect(backup_path)
try:
    backup_source.backup(backup_target)
finally:
    backup_target.close()
    backup_source.close()
backup_path.chmod(0o600)
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


def strict_config_json(raw: str | None) -> dict:
    try:
        value = json.loads(raw or "")
    except (TypeError, json.JSONDecodeError) as exc:
        raise SystemExit("Open WebUI legacy config JSON is invalid; refusing to overwrite it") from exc
    if not isinstance(value, dict):
        raise SystemExit("Open WebUI legacy config JSON is not an object; refusing to overwrite it")
    return value


existing_tables = tables()
changed_config = 0
changed_openai_connection = 0
config_schema = "missing"
changed_users = 0
activated_users = 0
changed_models = 0
changed_grants = 0
removed_models = 0
removed_grants = 0

obsolete_model_ids = {
    value
    for value in (custom_model_id,)
    if value and value != model_id
}

if "access_grant" in existing_tables and obsolete_model_ids:
    placeholders = ", ".join("?" for _ in obsolete_model_ids)
    cursor = con.execute(
        f"delete from access_grant where resource_type = 'model' and resource_id in ({placeholders})",
        tuple(obsolete_model_ids),
    )
    removed_grants += max(cursor.rowcount, 0)

if "model" in existing_tables and obsolete_model_ids:
    placeholders = ", ".join("?" for _ in obsolete_model_ids)
    cursor = con.execute(
        f"delete from model where id in ({placeholders})",
        tuple(obsolete_model_ids),
    )
    removed_models += max(cursor.rowcount, 0)

if "config" not in existing_tables:
    raise SystemExit("Open WebUI config table is missing")

config_columns = columns("config")
now = int(time.time())
if {"id", "data"}.issubset(config_columns):
    config_schema = "legacy"
    row = con.execute("select id, data from config order by id limit 1").fetchone()
    data = strict_config_json(row["data"]) if row else {}

    openai = data.get("openai")
    if not isinstance(openai, dict):
        openai = {}
        data["openai"] = openai
    openai["enable"] = True
    openai["api_base_urls"] = [openai_api_base_url]
    openai["api_keys"] = [openai_api_key]
    openai["api_configs"] = {}
    changed_openai_connection = 1

    data["default_model"] = model_id
    data["default_models"] = model_id
    data["default_pinned_models"] = model_id
    data["model_order_list"] = [model_id]
    data["title"] = model_name
    data["name"] = model_name
    data["default_prompt_suggestions"] = []
    ui = data.get("ui")
    if not isinstance(ui, dict):
        ui = {}
        data["ui"] = ui
    ui["default_model"] = model_id
    ui["default_models"] = model_id
    ui["default_pinned_models"] = model_id
    ui["model_order_list"] = [model_id]
    ui["prompt_suggestions"] = []
    evaluation = data.get("evaluation")
    if not isinstance(evaluation, dict):
        evaluation = {}
        data["evaluation"] = evaluation
    arena = evaluation.get("arena")
    if not isinstance(arena, dict):
        arena = {}
        evaluation["arena"] = arena
    arena["enable"] = False
    arena["models"] = []

    if row:
        query = "update config set data = ?"
        params = [json.dumps(data, ensure_ascii=False)]
        if "updated_at" in config_columns:
            query += ", updated_at = CURRENT_TIMESTAMP"
        query += " where id = ?"
        params.append(row["id"])
        con.execute(query, params)
    else:
        values = {
            "id": 1,
            "data": json.dumps(data, ensure_ascii=False),
            "version": 0,
        }
        parameter_columns = [name for name in values if name in config_columns]
        insert_columns = list(parameter_columns)
        insert_expressions = ["?" for _ in insert_columns]
        for timestamp_column in ("created_at", "updated_at"):
            if timestamp_column in config_columns:
                insert_columns.append(timestamp_column)
                insert_expressions.append("CURRENT_TIMESTAMP")
        con.execute(
            f"insert into config ({', '.join(insert_columns)}) values ({', '.join(insert_expressions)})",
            [values[name] for name in parameter_columns],
        )
    changed_config = 1
elif {"key", "value"}.issubset(config_columns):
    config_schema = "per_key"

    def read_key(key: str, default: object = None) -> object:
        row = con.execute("select value from config where key = ?", (key,)).fetchone()
        if not row:
            return default
        try:
            return json.loads(row["value"])
        except (TypeError, json.JSONDecodeError):
            return default

    def upsert_key(key: str, value: object) -> None:
        encoded = json.dumps(value, ensure_ascii=False)
        existing = con.execute("select key from config where key = ?", (key,)).fetchone()
        if existing:
            query = "update config set value = ?"
            params = [encoded]
            if "updated_at" in config_columns:
                query += ", updated_at = ?"
                params.append(now)
            query += " where key = ?"
            params.append(key)
            con.execute(query, params)
            return

        insert_values = {"key": key, "value": encoded, "updated_at": now}
        insert_columns = [name for name in insert_values if name in config_columns]
        placeholders = ", ".join("?" for _ in insert_columns)
        con.execute(
            f"insert into config ({', '.join(insert_columns)}) values ({placeholders})",
            [insert_values[name] for name in insert_columns],
        )

    for key, value in {
        "openai.enable": True,
        "openai.api_base_urls": [openai_api_base_url],
        "openai.api_keys": [openai_api_key],
        "openai.api_configs": {},
        "ui.default_models": model_id,
        "ui.default_pinned_models": model_id,
        "ui.model_order_list": [model_id],
        "ui.prompt_suggestions": [],
        "ollama.enable": False,
        "evaluation.arena.enable": False,
        "evaluation.arena.models": [],
    }.items():
        upsert_key(key, value)
    changed_openai_connection = 1
    changed_config = 1
else:
    raise SystemExit(
        "Unsupported Open WebUI config schema; expected id/data or key/value columns"
    )

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

    upsert_model(model_id, base_model_id=None, name=model_name)

if "access_grant" in existing_tables:
    grant_columns = columns("access_grant")
    grant_info = {
        row["name"]: row
        for row in con.execute("pragma table_info(access_grant)").fetchall()
    }

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
            "resource_type": "model",
            "resource_id": resource_id,
            "principal_type": "user",
            "principal_id": "*",
            "permission": "read",
            "created_at": int(time.time()),
        }
        id_info = grant_info.get("id")
        if id_info and "INT" not in (id_info["type"] or "").upper():
            values["id"] = str(uuid.uuid4())
        insert_columns = [column for column in values if column in grant_columns]
        placeholders = ", ".join("?" for _ in insert_columns)
        con.execute(
            f"insert into access_grant ({', '.join(insert_columns)}) values ({placeholders})",
            [values[column] for column in insert_columns],
        )
        changed_grants += 1

    ensure_public_read(model_id)

if "user" in existing_tables:
    user_columns = columns("user")
    if {"id", "settings"}.issubset(user_columns):
        rows = con.execute("select id, settings, role from user" if "role" in user_columns else "select id, settings from user").fetchall()
        for row in rows:
            settings = safe_json(row["settings"])
            settings["default_model"] = model_id
            settings["default_models"] = model_id
            settings["default_pinned_models"] = model_id
            settings["model_order_list"] = [model_id]
            ui = settings.setdefault("ui", {})
            ui["default_model"] = model_id
            ui["default_models"] = model_id
            ui["default_pinned_models"] = model_id
            ui["model_order_list"] = [model_id]
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

if config_schema == "legacy":
    verified_row = con.execute("select data from config order by id limit 1").fetchone()
    verified_data = strict_config_json(verified_row["data"] if verified_row else None)
    verified_openai = verified_data.get("openai")
    connection_verified = (
        isinstance(verified_openai, dict)
        and verified_openai.get("enable") is True
        and verified_openai.get("api_base_urls") == [openai_api_base_url]
        and verified_openai.get("api_keys") == [openai_api_key]
    )
else:
    connection_verified = (
        read_key("openai.enable") is True
        and read_key("openai.api_base_urls") == [openai_api_base_url]
        and read_key("openai.api_keys") == [openai_api_key]
    )

if not connection_verified:
    raise SystemExit("Open WebUI persistent OpenAI connection verification failed")

con.commit()

model_rows = []
if "model" in existing_tables:
    selected_columns = [name for name in ("id", "base_model_id", "name", "is_active") if name in columns("model")]
    if selected_columns:
        model_rows = [dict(row) for row in con.execute(f"select {', '.join(selected_columns)} from model order by id").fetchall()]

grant_rows = []
if "access_grant" in existing_tables:
    selected_columns = [
        name
        for name in ("resource_type", "resource_id", "principal_type", "principal_id", "permission")
        if name in columns("access_grant")
    ]
    if selected_columns:
        grant_rows = [
            dict(row)
            for row in con.execute(
                f"select {', '.join(selected_columns)} from access_grant where resource_type = 'model' order by resource_id"
            ).fetchall()
        ]

con.close()

print(f"config_schema={config_schema}")
print(f"config_updated={changed_config}")
print(f"openai_connection_updated={changed_openai_connection}")
print("openai_connection_verified=1")
if verify_upstream:
    print(f"upstream_model_verified={model_id}")
print(f"models_updated={changed_models}")
print(f"models_removed={removed_models}")
print(f"public_grants_added={changed_grants}")
print(f"obsolete_grants_removed={removed_grants}")
print(f"users_updated={changed_users}")
print(f"pending_activated={activated_users}")
print("model_rows=" + json.dumps(model_rows, ensure_ascii=False))
print("model_grants=" + json.dumps(grant_rows, ensure_ascii=False))
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
                    "verify_upstream": True,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        run(["docker", "cp", str(inner), f"{args.container}:/tmp/{inner.name}"])
        run(["docker", "cp", str(config), f"{args.container}:/tmp/{config.name}"])
        command = ["docker", "exec"]
        for name in ("OPENAI_COMPAT_API_KEY", "OPENAI_COMPAT_API_BASE_URL"):
            if os.environ.get(name):
                command.extend(["-e", name])
        command.extend(
            [args.container, "python", f"/tmp/{inner.name}", f"/tmp/{config.name}"]
        )
        run(command)


if __name__ == "__main__":
    main()
