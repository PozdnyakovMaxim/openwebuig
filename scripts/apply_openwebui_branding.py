#!/usr/bin/env python3
"""Apply branding to a running chat UI container."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import shlex
import subprocess
import tempfile
import time
from pathlib import Path


STATIC_DIRS = (
    "/app/build/static",
    "/app/backend/open_webui/static",
)


CUSTOM_CSS = """
:root {
  --project-brand-blue: #1468ff;
}

img[src*="brand-logo.svg"],
img[src*="brand-icon.svg"],
img[src*="favicon.svg"],
img[src*="logo.png"],
img[src*="logo.PNG"],
img[src*="/models/model/profile/image"] {
  border-radius: 4px !important;
  object-fit: contain !important;
  border-color: transparent !important;
}

img[src*="brand-splash.svg"],
img[src*="splash"] {
  border-radius: 22px !important;
  object-fit: contain !important;
}

a[href*="open-webui"],
a[href*="openwebui"],
a[href*="github.com/open-webui"],
a[href*="docs.openwebui"],
a[href*="discord.gg/open-webui"],
a[href*="releases/tag"],
a[href*="changelog"] {
  display: none !important;
}

[data-testid*="changelog"],
[data-testid*="version"],
[data-testid*="release"] {
  display: none !important;
}

[data-sonner-toaster],
[data-sonner-toast],
section[aria-label="Notifications"],
div[role="status"],
div[role="alert"] {
  display: none !important;
  opacity: 0 !important;
  pointer-events: none !important;
}

button[aria-label="Set as default"],
button[title="Set as default"] {
  display: none !important;
}

form + div[class*="text-xs"],
form + button[class*="text-xs"],
main div[class*="text-xs"][class*="text-gray-500"]:has(a[href*="github"]),
main div[class*="text-xs"][class*="text-gray-500"]:has(a[href*="releases"]) {
  display: none !important;
}

img[alt="Open WebUI"],
img[alt="open-webui"],
img[title="Open WebUI"],
svg[aria-label="Open WebUI"] {
  content: url("/static/brand-icon.svg") !important;
}

footer,
[data-testid*="footer"],
[class*="changelog"],
[class*="version"] {
  display: none !important;
}
"""


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    print("+", shlex.join(command))
    return subprocess.run(command, check=True, text=True)


def docker_exec(container: str, command: str) -> None:
    run(["docker", "exec", container, "sh", "-lc", command])


def make_logo_mark_svg(
    image_path: Path,
    output_path: Path,
    *,
    width: int,
    height: int,
    radius: int,
) -> None:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="{width}" height="{height}" rx="{radius}" ry="{radius}" fill="#fff"/>
  <image href="data:{mime_type};base64,{encoded}" x="0" y="0" width="{width}" height="{height}" preserveAspectRatio="xMidYMid meet"/>
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")


def write_patcher(path: Path) -> None:
    patcher = r'''
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
brand_name = config["brand_name"]
technical_label = config["technical_label"]
model_id = config["model_id"]
default_model_id = config.get("default_model_id") or model_id

runtime_script = f"""
<script id="brand-runtime-patch">
(() => {{
  const brandName = {json.dumps(config["brand_name"], ensure_ascii=False)};
  const brandIcon = "/static/brand-icon.svg";
  const openWebUiPattern = /Open\\s*WebUI|OpenWebUI|OPEN\\s*WEBUI/i;
  const openWebUiReplacePattern = /Open\\s*WebUI|OpenWebUI|OPEN\\s*WEBUI/g;

  function patchTitle() {{
    if (!document.title || openWebUiPattern.test(document.title)) {{
      document.title = document.title.replace(openWebUiReplacePattern, brandName) || brandName;
    }}
  }}

  function patchTextNode(node) {{
    if (!node || node.nodeType !== Node.TEXT_NODE || !node.nodeValue) {{
      return;
    }}
    const value = node.nodeValue;
    const trimmed = value.trim();
    const greeting = trimmed.match(/^(?:Hello|Hi|Hey),\s+(.+)$/i);
    if (greeting) {{
      const parts = greeting[1].trim().split(/\s+/);
      const firstName = parts.length >= 2 ? parts[1] : parts[0];
      node.nodeValue = `Приветствую, ${{firstName}}`;
      return;
    }}
    if (trimmed === "oi" || trimmed === "OI") {{
      const img = document.createElement("img");
      img.src = brandIcon;
      img.alt = brandName;
      img.style.width = "1.25rem";
      img.style.height = "1.25rem";
      img.style.objectFit = "contain";
      node.replaceWith(img);
      return;
    }}
    if (openWebUiPattern.test(value)) {{
      node.nodeValue = value.replace(openWebUiReplacePattern, brandName);
    }}
  }}

  function patchAttributes() {{
    document.querySelectorAll("*").forEach((element) => {{
      ["aria-label", "title", "placeholder", "alt", "data-title"].forEach((attr) => {{
        const value = element.getAttribute?.(attr);
        if (value && openWebUiPattern.test(value)) {{
          element.setAttribute(attr, value.replace(openWebUiReplacePattern, brandName));
        }}
      }});
    }});
  }}

  function patchImages() {{
    document.querySelectorAll("img").forEach((img) => {{
      const src = img.getAttribute("src") || "";
      const alt = img.getAttribute("alt") || "";
      const title = img.getAttribute("title") || "";
      if (
        src.includes("open-webui") ||
        src.includes("favicon") ||
        src.includes("logo") ||
        src.includes("/models/model/profile/image") ||
        alt.match(openWebUiPattern) ||
        title.match(openWebUiPattern) ||
        (img.complete && img.naturalWidth === 0)
      ) {{
        img.src = brandIcon;
        img.alt = brandName;
        img.title = brandName;
        img.style.objectFit = "contain";
      }}
    }});
  }}

  function patchFavicon() {{
    document.querySelectorAll('link[rel*="icon"]').forEach((link) => {{
      link.setAttribute("href", brandIcon);
      link.setAttribute("type", "image/svg+xml");
    }});
  }}

  function patchPage() {{
    patchTitle();
    patchFavicon();
    patchAttributes();
    patchImages();
    const walker = document.createTreeWalker(document.body || document.documentElement, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (nodes.length < 5000) {{
      const node = walker.nextNode();
      if (!node) break;
      nodes.push(node);
    }}
    nodes.forEach(patchTextNode);
    document.querySelectorAll("a[href*='open-webui'], a[href*='openwebui'], a[href*='github.com/open-webui']").forEach((node) => {{
      node.style.display = "none";
    }});
  }}

  window.addEventListener("error", (event) => {{
    const target = event.target;
    if (target instanceof HTMLImageElement) {{
      target.src = brandIcon;
      target.alt = brandName;
    }}
  }}, true);

  const observer = new MutationObserver(() => {{
    window.clearTimeout(window.__brandPatchTimer);
    window.__brandPatchTimer = window.setTimeout(patchPage, 60);
  }});

  if (document.readyState === "loading") {{
    document.addEventListener("DOMContentLoaded", () => {{
      patchPage();
      observer.observe(document.body, {{ childList: true, subtree: true, characterData: true }});
    }});
  }} else {{
    patchPage();
    observer.observe(document.body, {{ childList: true, subtree: true, characterData: true }});
  }}

  let attempts = 0;
  const interval = window.setInterval(() => {{
    patchPage();
    attempts += 1;
    if (attempts >= 20) {{
      window.clearInterval(interval);
    }}
  }}, 500);
}})();
</script>
"""

replacements = {
    "Open WebUI Community": brand_name,
    "Войти в Open WebUI по LDAP": f"Войти в {brand_name} по LDAP",
    "Sign in to Open WebUI with LDAP": f"Войти в {brand_name} по LDAP",
    "Login to Open WebUI with LDAP": f"Войти в {brand_name} по LDAP",
    "Open WebUI по LDAP": f"{brand_name} по LDAP",
    "Open WebUI": brand_name,
    "OpenWebUI": brand_name,
    "OPEN WEBUI": brand_name.upper(),
    " (Open WebUI)": "",
    "open-webui": technical_label,
    "openwebui": technical_label,
    "docs.openwebui.com": "#",
    "https://openwebui.com": "#",
    "https://github.com/open-webui/open-webui": "#",
    "https://github.com/open-webui": "#",
    "/static/logo.png": "/static/brand-icon.svg",
    "static/logo.png": "static/brand-icon.svg",
    "logo.png": "brand-icon.svg",
    "brand-logo.svg": "brand-icon.svg",
    "/static/splash.png": "/static/brand-splash.svg",
    "static/splash.png": "static/brand-splash.svg",
    "splash.png": "brand-splash.svg",
    "/static/splash-dark.png": "/static/brand-splash.svg",
    "static/splash-dark.png": "static/brand-splash.svg",
    "splash-dark.png": "brand-splash.svg",
    "/static/favicon.png": "/static/favicon.svg",
    "static/favicon.png": "static/favicon.svg",
    "favicon.png": "favicon.svg",
    "web-app-manifest-192x192.png": "brand-icon.svg",
    "web-app-manifest-512x512.png": "brand-icon.svg",
    "apple-touch-icon.png": "brand-icon.svg",
    "What's New": "",
    "Release Notes": "",
    "Changelog": "",
    "Update Available": "",
    "Open Sidebar": "Открыть боковое меню",
    "Close Sidebar": "Закрыть боковое меню",
    "New Chat": "Новый чат",
    "Search In Models": "Поиск по моделям",
    "Search": "Поиск",
    "Notes": "Заметки",
    "Workspace": "Рабочая область",
    "Folders": "Папки",
    "Chats": "Чаты",
    "Open User Profile Menu": "Открыть меню пользователя",
    "User Menu": "Меню пользователя",
    "User menu": "Меню пользователя",
    "Controls": "Параметры",
    "Settings": "Настройки",
    "Voice Input": "Голосовой ввод",
    "Voice mode": "Голосовой режим",
    "Add Model": "Добавить модель",
    "Remove Model": "Удалить модель",
    "Select a model": "Выберите модель",
    "Selected model: {{modelName}}": "Выбранная модель: {{modelName}}",
    "{{modelName}} profile image": "Изображение модели {{modelName}}",
    "Get information on {{name}} in the UI": "Информация о {{name}}",
    "Models": "Модели",
    "Channels": "Каналы",
    "Create Channel": "Создать канал",
    "New Folder": "Новая папка",
    "Pinned": "Закрепленные",
    "Direct": "Напрямую",
    "External": "Внешняя",
    "Temporary Chat": "Временный чат",
    "This chat won't appear in history and your messages will not be saved.": "Этот чат не появится в истории, сообщения не будут сохранены.",
    "More Options": "Еще",
    "Copied link to clipboard": "Ссылка скопирована",
    "Failed to copy link": "Не удалось скопировать ссылку",
    "Loading...": "Загрузка...",
    "User": "Пользователь",
    "How can I help you today?": "Задайте вопрос по документам",
    "Suggested": "",
    "Set as default": "",
    "You're now logged in.": "",
    "You're now logged in": "",
    "You are now logged in.": "",
    "Default model updated": "",
}

version_patterns = [
    re.compile(rf"{re.escape(brand_name)}\s*·\s*v\d+(?:\.\d+)+"),
    re.compile(rf"{re.escape(brand_name)}\s*‧\s*v\d+(?:\.\d+)+"),
    re.compile(rf"{re.escape(brand_name)}\s*-\s*v\d+(?:\.\d+)+"),
    re.compile(r"Open WebUI\s*·\s*v\d+(?:\.\d+)+"),
    re.compile(r"Open WebUI\s*-\s*v\d+(?:\.\d+)+"),
    re.compile(r"j\((?P<target>[A-Za-z_$][\w$]*),`\$\{[^`]*?\}\s*[·‧]\s*v\$\{[^`]*?\}`\)"),
]

roots = [Path("/app/build"), Path("/app/backend/open_webui")]
suffixes = {
    ".css",
    ".html",
    ".js",
    ".json",
    ".svg",
    ".webmanifest",
    ".xml",
}

patched = 0
for root in roots:
    if not root.exists():
        continue
    for file_path in root.rglob("*"):
        if not file_path.is_file() or file_path.suffix not in suffixes:
            continue
        try:
            original = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        updated = original
        for old, new in replacements.items():
            updated = updated.replace(old, new)
        for pattern in version_patterns:
            if "target" in pattern.groupindex:
                updated = pattern.sub(lambda match: f'j({match.group("target")},"")', updated)
            else:
                updated = pattern.sub("", updated)
        if updated != original:
            file_path.write_text(updated, encoding="utf-8")
            patched += 1

print(f"patched_text_files={patched}")

html_patched = 0
for html_file in (Path("/app/build/index.html"), Path("/app/backend/open_webui/static/index.html")):
    if not html_file.exists():
        continue
    original = html_file.read_text(encoding="utf-8")
    updated = re.sub(
        r"\n?<script id=\"brand-runtime-patch\">.*?</script>",
        "",
        original,
        flags=re.DOTALL,
    )
    if "</body>" in updated:
        updated = updated.replace("</body>", runtime_script + "\n</body>")
    elif "</head>" in updated:
        updated = updated.replace("</head>", runtime_script + "\n</head>")
    else:
        updated += runtime_script
    if updated != original:
        html_file.write_text(updated, encoding="utf-8")
        html_patched += 1

print(f"patched_runtime_html={html_patched}")

env_file = Path("/app/backend/open_webui/env.py")
if env_file.exists():
    original = env_file.read_text(encoding="utf-8")
    updated = re.sub(
        r'WEBUI_NAME = os\.environ\.get\("WEBUI_NAME", .*?\)',
        f'WEBUI_NAME = os.environ.get("WEBUI_NAME", {brand_name!r})',
        original,
    )
    updated = re.sub(
        r'if\s+WEBUI_NAME\s*!=\s*(["\'])Open WebUI\1:\s*\n\s*WEBUI_NAME\s*\+=\s*(["\'])\s*\(Open WebUI\)\2',
        'if False:\n    WEBUI_NAME += ""',
        updated,
    )
    updated = updated.replace('WEBUI_NAME += " (Open WebUI)"', 'WEBUI_NAME += ""')
    updated = updated.replace("WEBUI_NAME += ' (Open WebUI)'", "WEBUI_NAME += ''")
    updated = updated.replace(
        'WEBUI_FAVICON_URL = "https://openwebui.com/favicon.png"',
        'WEBUI_FAVICON_URL = "/static/favicon.svg"',
    )
    if updated != original:
        env_file.write_text(updated, encoding="utf-8")
        print("patched_runtime_env=true")

main_file = Path("/app/backend/open_webui/main.py")
if main_file.exists():
    original = main_file.read_text(encoding="utf-8")
    updated = re.sub(
        r'title=(["\'])(?:Open WebUI|ГлавстройLLM)\1',
        f"title={brand_name!r}",
        original,
    )
    if updated != original:
        main_file.write_text(updated, encoding="utf-8")
        print("patched_runtime_main=true")

models_router_file = Path("/app/backend/open_webui/routers/models.py")
if models_router_file.exists():
    original = models_router_file.read_text(encoding="utf-8")
    updated = original.replace(
        'return FileResponse(f"{STATIC_DIR}/favicon.png")',
        'return FileResponse(f"{STATIC_DIR}/brand-icon.svg", media_type="image/svg+xml")',
    )
    if updated != original:
        models_router_file.write_text(updated, encoding="utf-8")
        print("patched_model_profile_image=true")

db_file = Path("/app/backend/data/webui.db")
if db_file.exists():
    import sqlite3

    con = sqlite3.connect(db_file)
    try:
        def table_columns(table_name: str) -> set[str]:
            return {row[1] for row in con.execute(f"pragma table_info({table_name})").fetchall()}

        row = con.execute("select id, data from config order by id limit 1").fetchone()
        if row:
            config_id, raw = row
            try:
                data = json.loads(raw)
            except Exception:
                data = {}
            ui = data.setdefault("ui", {})
            data["name"] = brand_name
            data["title"] = brand_name
            data["default_models"] = default_model_id
            data["default_pinned_models"] = default_model_id
            data["default_model"] = default_model_id
            data["default_prompt_suggestions"] = []
            ui["prompt_suggestions"] = []
            ui["enable_signup"] = False
            ui["locale"] = "ru-RU"
            ui["language"] = "ru-RU"
            ui["default_models"] = default_model_id
            ui["default_pinned_models"] = default_model_id
            ui["default_model"] = default_model_id
            ui["show_changelog"] = False
            evaluation = data.setdefault("evaluation", {})
            arena = evaluation.setdefault("arena", {})
            arena["enable"] = False
            arena["models"] = []
            con.execute(
                "update config set data = ?, updated_at = datetime('now') where id = ?",
                (json.dumps(data, ensure_ascii=False), config_id),
            )
        else:
            data = {
                "version": 0,
                "name": brand_name,
                "title": brand_name,
                "default_models": default_model_id,
                "default_pinned_models": default_model_id,
                "default_model": default_model_id,
                "ui": {
                    "enable_signup": False,
                    "prompt_suggestions": [],
                    "default_models": default_model_id,
                    "default_pinned_models": default_model_id,
                    "default_model": default_model_id,
                    "show_changelog": False,
                },
                "evaluation": {"arena": {"enable": False, "models": []}},
            }
            con.execute(
                "insert into config (id, data, version, created_at, updated_at) values (1, ?, 0, datetime('now'), datetime('now'))",
                (json.dumps(data, ensure_ascii=False),),
            )
        for user_id, raw_settings in con.execute("select id, settings from user").fetchall():
            try:
                settings = json.loads(raw_settings or "{}")
            except Exception:
                settings = {}
            ui = settings.setdefault("ui", {})
            ui["locale"] = "ru-RU"
            ui["language"] = "ru-RU"
            ui["default_models"] = default_model_id
            ui["default_pinned_models"] = default_model_id
            ui["default_model"] = default_model_id
            con.execute(
                "update user set settings = ?, updated_at = strftime('%s', 'now') where id = ?",
                (json.dumps(settings, ensure_ascii=False), user_id),
            )
        tables = {row[0] for row in con.execute("select name from sqlite_master where type='table'").fetchall()}
        if "model" in tables:
            columns = table_columns("model")
            now = int(time.time())
            user_row = con.execute("select id from user order by created_at limit 1").fetchone()
            user_id = user_row[0] if user_row else "system"
            meta = {
                "profile_image_url": "/static/brand-icon.svg",
                "description": "Поиск и ответы по документам",
                "capabilities": {"vision": False, "citations": True},
            }
            params = {}
            existing = con.execute("select id from model where id = ?", (model_id,)).fetchone()
            if existing:
                assignments = []
                values = []
                for name, value in {
                    "name": brand_name,
                    "meta": json.dumps(meta, ensure_ascii=False),
                    "params": json.dumps(params, ensure_ascii=False),
                    "is_active": 1,
                    "updated_at": now,
                }.items():
                    if name in columns:
                        assignments.append(f"{name} = ?")
                        values.append(value)
                if assignments:
                    values.append(model_id)
                    con.execute(f"update model set {', '.join(assignments)} where id = ?", values)
            else:
                payload = {
                    "id": model_id,
                    "user_id": user_id,
                    "base_model_id": None,
                    "name": brand_name,
                    "params": json.dumps(params, ensure_ascii=False),
                    "meta": json.dumps(meta, ensure_ascii=False),
                    "access_control": None,
                    "is_active": 1,
                    "created_at": now,
                    "updated_at": now,
                }
                insert_columns = [name for name in payload if name in columns]
                placeholders = ", ".join("?" for _ in insert_columns)
                con.execute(
                    f"insert into model ({', '.join(insert_columns)}) values ({placeholders})",
                    [payload[name] for name in insert_columns],
                )
            print("patched_model_profile=true")
        con.commit()
        print("patched_runtime_db=true")
    finally:
        con.close()
'''
    path.write_text(patcher, encoding="utf-8")


def append_custom_css(container: str, css_file_name: str) -> None:
    for static_dir in STATIC_DIRS:
        target = f"{static_dir}/custom.css"
        docker_exec(
            container,
            f"cat /tmp/{shlex.quote(css_file_name)} > {shlex.quote(target)}",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Brand a running Open WebUI container without rebuilding the image."
    )
    parser.add_argument("--container", default="document-search-webui")
    parser.add_argument("--logo", required=True, help="Path to a PNG/JPG logo image.")
    parser.add_argument("--brand-name", default="ГлавстройLLM")
    parser.add_argument("--model-id", default="document-search-rag")
    parser.add_argument("--default-model-id", default=None)
    parser.add_argument(
        "--technical-label",
        default="project-ui",
        help="ASCII replacement for internal open-webui slugs and URLs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logo_path = Path(args.logo).expanduser().resolve()
    if not logo_path.exists():
        raise SystemExit(f"Logo file does not exist: {logo_path}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        brand_logo = tmp_path / "brand-logo.svg"
        brand_splash = tmp_path / "brand-splash.svg"
        brand_icon = tmp_path / "brand-icon.svg"
        favicon = tmp_path / "favicon.svg"
        custom_css = tmp_path / "custom.css"
        patcher = tmp_path / "patch_openwebui_branding.py"
        config = tmp_path / "branding_config.json"

        make_logo_mark_svg(logo_path, brand_logo, width=512, height=512, radius=0)
        make_logo_mark_svg(logo_path, brand_splash, width=512, height=512, radius=0)
        make_logo_mark_svg(logo_path, brand_icon, width=512, height=512, radius=0)
        make_logo_mark_svg(logo_path, favicon, width=512, height=512, radius=0)
        custom_css.write_text(CUSTOM_CSS, encoding="utf-8")
        write_patcher(patcher)
        config.write_text(
            json.dumps(
                {
                    "brand_name": args.brand_name,
                    "technical_label": args.technical_label,
                    "model_id": args.model_id,
                    "default_model_id": args.default_model_id,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        docker_exec(args.container, "mkdir -p " + " ".join(shlex.quote(d) for d in STATIC_DIRS))
        for static_dir in STATIC_DIRS:
            for asset in (brand_logo, brand_splash, brand_icon, favicon):
                run(["docker", "cp", str(asset), f"{args.container}:{static_dir}/{asset.name}"])

        run(["docker", "cp", str(custom_css), f"{args.container}:/tmp/{custom_css.name}"])
        run(["docker", "cp", str(patcher), f"{args.container}:/tmp/{patcher.name}"])
        run(["docker", "cp", str(config), f"{args.container}:/tmp/{config.name}"])

        docker_exec(
            args.container,
            f"python /tmp/{shlex.quote(patcher.name)} /tmp/{shlex.quote(config.name)}",
        )
        append_custom_css(args.container, custom_css.name)

    print("branding_applied=true")


if __name__ == "__main__":
    main()
