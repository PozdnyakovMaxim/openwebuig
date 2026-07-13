# Document Search RAG

Инструкция по запуску RAG API и Open WebUI на Ubuntu.

## 1. Что должно быть на сервере

Рабочая схема:

```text
~/openwebuig
~/rag_template
Docker container: document-search-postgres
Docker container: document-search-webui
RAG API: http://127.0.0.1:8000
Open WebUI через nginx: http://SERVER_IP
Open WebUI напрямую: http://SERVER_IP:3000
```

`~/openwebuig` содержит код API и скрипты.

`~/rag_template` может оставаться старой рабочей папкой с документами и индексом, если `.env` в `~/openwebuig` указывает на ту же базу.

Документы, `.env`, `.venv`, `artifacts` и реальные выгрузки в GitHub не загружать.

## 2. Быстрый запуск после перерыва

Открой первый терминал.

```bash
sudo docker start document-search-postgres
```

```bash
cd ~/openwebuig
```

```bash
uv run python scripts/serve_openai_compatible.py --host 0.0.0.0 --port 8000
```

Этот терминал не закрывать.

Открой второй терминал.

```bash
sudo docker start document-search-webui
```

Проверить API:

```bash
curl http://127.0.0.1:8000/v1/models
```

Проверить Open WebUI:

```bash
curl http://127.0.0.1:3000
```

Открыть в браузере:

```text
http://SERVER_IP
```

Для текущего стенда:

```text
http://172.19.225.124
```

## 3. Запуск без привязки к открытому терминалу

Если нужно закрывать терминал, запускай RAG API через `nohup`:

```bash
cd ~/openwebuig
```

```bash
nohup uv run python scripts/serve_openai_compatible.py --host 0.0.0.0 --port 8000 > rag-api.log 2>&1 &
```

Проверить:

```bash
curl http://127.0.0.1:8000/v1/models
```

Посмотреть лог:

```bash
tail -f ~/openwebuig/rag-api.log
```

Остановить такой запуск:

```bash
pkill -f serve_openai_compatible.py
```

## 4. Правильный постоянный запуск через systemd

Для тестового контура лучше использовать `systemd`, чтобы RAG API сам поднимался после перезагрузки.

Создать сервис:

```bash
sudo nano /etc/systemd/system/document-search-rag-api.service
```

Вставить:

```ini
[Unit]
Description=Document Search RAG API
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=simple
User=pozdniakov
WorkingDirectory=/home/pozdniakov/openwebuig
ExecStart=/home/pozdniakov/.local/bin/uv run python scripts/serve_openai_compatible.py --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Если `uv` лежит в другом месте, проверить:

```bash
which uv
```

И заменить путь в `ExecStart`.

Включить сервис:

```bash
sudo systemctl daemon-reload
```

```bash
sudo systemctl enable document-search-rag-api
```

```bash
sudo systemctl start document-search-rag-api
```

Проверить:

```bash
sudo systemctl status document-search-rag-api
```

```bash
curl http://127.0.0.1:8000/v1/models
```

Логи:

```bash
journalctl -u document-search-rag-api -f
```

После этого отдельный терминал для RAG API больше не нужен.

## 5. Open WebUI container

Если контейнер уже существует:

```bash
sudo docker start document-search-webui
```

Если контейнер нужно создать заново, volume не удалять:

```bash
sudo docker rm -f document-search-webui
```

```bash
sudo docker run -d --name document-search-webui --restart unless-stopped -p 3000:8080 -v open-webui:/app/backend/data --add-host host.docker.internal:host-gateway -e ENABLE_OPENAI_API=True -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 -e OPENAI_API_KEY=anything -e OPENAI_API_BASE_URLS=http://host.docker.internal:8000/v1 -e OPENAI_API_KEYS=anything -e OFFLINE_MODE=True -e HF_HUB_OFFLINE=1 -e ENABLE_VERSION_UPDATE_CHECK=False ghcr.io/open-webui/open-webui:main
```

Не использовать `RESET_CONFIG_ON_START=true` без необходимости. Он может сбрасывать настройки Open WebUI при рестарте.

Не выполнять:

```bash
sudo docker volume rm open-webui
```

В этом volume лежат пользователи, настройки и чаты Open WebUI.

## 6. Проверки

Проверить контейнеры:

```bash
sudo docker ps
```

Проверить RAG API:

```bash
curl http://127.0.0.1:8000/v1/models
```

Проверить, что Open WebUI видит RAG API:

```bash
sudo docker exec document-search-webui python -c "import urllib.request;print(urllib.request.urlopen('http://host.docker.internal:8000/v1/models',timeout=5).read().decode())"
```

Проверить Open WebUI:

```bash
curl http://127.0.0.1:3000
```

## 6.1. Замер скорости RAG и API модели

После обновления RAG API возвращает длительность этапов в `Server-Timing`, `X-RAG-Total-Ms` и в поле `rag_metrics` обычного JSON-ответа. В журнал systemd записываются маршрут и длительности без полного текста вопроса.

Измерить полный RAG, включая BGE-M3, pgvector и генерацию Qwen:

```bash
cd ~/openwebuig
uv run python scripts/benchmark_latency.py "Какие требования установлены к резервному копированию?" --target rag --runs 5
```

Измерить только корпоративный Qwen API без BGE-M3 и pgvector:

```bash
cd ~/openwebuig
uv run python scripts/benchmark_latency.py "Кратко опиши назначение ИТ-поддержки" --target provider --runs 5
```

Перед измерением выполняется один прогревочный запрос. В результате выводятся `min`, `avg`, `p95` и `max`. Для RAG дополнительно выводятся `embedding_ms`, `search_ms` и `generation_ms`.

Посмотреть метрики живых запросов Open WebUI:

```bash
sudo journalctl -u document-search-rag-api -f
```

## 7. Если модель пропала в Open WebUI

Сначала проверить RAG API:

```bash
curl http://127.0.0.1:8000/v1/models
```

Если ответа нет, запустить RAG API:

```bash
cd ~/openwebuig
```

```bash
uv run python scripts/serve_openai_compatible.py --host 0.0.0.0 --port 8000
```

Если ответ есть, проверить из контейнера Open WebUI:

```bash
sudo docker exec document-search-webui python -c "import urllib.request;print(urllib.request.urlopen('http://host.docker.internal:8000/v1/models',timeout=5).read().decode())"
```

Если контейнер видит модель, зайти в Open WebUI:

```text
Admin Panel -> Settings -> Connections
```

Проверить connection:

```text
http://host.docker.internal:8000/v1
```

API key:

```text
anything
```

Сохранить, обновить страницу `Ctrl+F5`.

Если у администратора модель есть, а у обычного или LDAP-пользователя список моделей пустой:

```bash
cd ~/openwebuig
```

```bash
sudo bash scripts/recreate_openwebui_for_ldap_users.sh
```

Скрипт сохраняет volume `open-webui`, поэтому пользователи и чаты остаются. Он пересоздает только контейнер, включает доступ к единственному внутреннему RAG API для всех прошедших LDAP-аутентификацию пользователей, отключает `Arena Model`, оставляет одну модель `ГлавстройLLM` и повторно применяет брендинг.

После этого выйти из Open WebUI, войти заново и обновить страницу `Ctrl+F5`.

Если пользователь остается в статусе ожидания активации, можно дополнительно активировать pending-пользователей:

```bash
sudo python3 scripts/fix_openwebui_model_access.py --container document-search-webui --model-id document-search-rag --custom-model-id glavstroy-llm --model-name ГлавстройLLM --activate-pending
```

Скрипт перед изменениями автоматически создает backup `webui.db` внутри volume Open WebUI. Он удаляет старый дубль `glavstroy-llm`, оставляет одну модель `document-search-rag` с именем `ГлавстройLLM`, назначает ее моделью по умолчанию и выдает публичный `read`-доступ через `access_grant`.

## 8. Если чаты пропали слева

Сначала проверить, что вошел тот же пользователь.

Проверить, что контейнер использует правильный volume:

```bash
sudo docker inspect document-search-webui | grep -A 20 Mounts
```

Должен быть volume:

```text
open-webui:/app/backend/data
```

Проверить количество чатов:

```bash
sudo docker exec document-search-webui python -c "import sqlite3; con=sqlite3.connect('/app/backend/data/webui.db'); print(con.execute('select count(*) from chat').fetchone())"
```

Если число больше `0`, чаты физически есть. Обычно проблема в пользователе или сессии.

## 9. Брендинг

Если контейнер пересоздавали или обновляли образ, брендинг нужно применить заново.

```bash
cd ~/openwebuig
```

Проверить логотип:

```bash
ls -la branding
```

Основной файл логотипа называется `branding/logo.PNG`:

```bash
sudo python3 scripts/apply_openwebui_branding.py --container document-search-webui --logo branding/logo.PNG --brand-name ГлавстройLLM --model-id document-search-rag --default-model-id document-search-rag
```

Если используется старый SVG:

```bash
sudo python3 scripts/apply_openwebui_branding.py --container document-search-webui --logo branding/logo.svg --brand-name ГлавстройLLM --model-id document-search-rag
```

После брендинга:

```bash
sudo docker restart document-search-webui
```

Обычный LDAP-вход:

```text
http://SERVER_IP/auth
```

Служебный Email-вход администратора:

```text
http://SERVER_IP/auth?admin=1
```

На обычной странице ссылка Email скрыта. Параметр `admin=1` показывает кнопку `Войти по Email`.

Проверить, что Open WebUI отдает страницу через nginx:

```bash
curl http://127.0.0.1 | head
```

В браузере:

```text
Ctrl+F5
```

Скрипт меняет название вкладки, LDAP-экран входа, маленькие `oi`-иконки, аватар модели, favicon, footer с версией и модель по умолчанию для пользователей.

Если в консоли браузера появилась ошибка `URLUnlockParams is not defined` или `URLПоискParams is not defined`, собранный JavaScript был поврежден старой версией скрипта брендинга. Обычный рестарт такой контейнер не восстановит. Обновить репозиторий и пересоздать только контейнер:

```bash
cd ~/openwebuig
git pull
sudo bash scripts/recreate_openwebui_for_ldap_users.sh
```

Скрипт берет чистые файлы из уже установленного Docker-образа и повторно подключает существующий volume `open-webui`. Пользователи, LDAP-настройки и история чатов сохраняются. Новая версия брендинга не изменяет собранные `.js`-файлы.

Если после рестарта модель пропала, проверить RAG API на `8000`. Частая причина: был закрыт терминал, где работал RAG API.

## 10. Обновление кода из GitHub

```bash
cd ~/openwebuig
```

```bash
git pull
```

Если менялся Python-код API, перезапустить RAG API.

При ручном запуске:

```text
Ctrl+C
```

Потом:

```bash
uv run python scripts/serve_openai_compatible.py --host 0.0.0.0 --port 8000
```

При `systemd`:

```bash
sudo systemctl restart document-search-rag-api
```

Open WebUI перезапускать не нужно, если менялся только RAG API.

## 11. Переиндексация документов

Если документы обновились:

```bash
cd ~/openwebuig
```

```bash
uv run python scripts/step1_extract_corpus.py --input "docs/**/*.docx" --output-dir artifacts/extracted
```

```bash
uv run python scripts/step1_validate_extraction.py --input-dir artifacts/extracted
```

```bash
uv run python scripts/step2_chunk_corpus.py --input-dir artifacts/extracted --output-dir artifacts/chunks
```

```bash
uv run python scripts/step2_validate_chunks.py --input-dir artifacts/chunks
```

```bash
uv run python scripts/step3_init_pgvector.py --embedding-dim 1024
```

```bash
uv run python scripts/step3_index_chunks.py --input-dir artifacts/chunks
```

```bash
uv run python scripts/step3_validate_index.py --query "резервное копирование"
```

Не использовать `--recreate`, если не нужно полностью удалить старый индекс.

## 12. Переменные окружения

Файл:

```text
~/openwebuig/.env
```

Важное:

```text
DATABASE_URL=postgresql://postgres:PASSWORD@127.0.0.1:5432/document_search
EMBEDDING_PROVIDER=local
CHAT_PROVIDER=provider
LOCAL_EMBED_MODEL=/opt/models/bge-m3
PROVIDER_API_BASE_URL=https://provider-host/v1
PROVIDER_API_KEY=...
PROVIDER_CHAT_MODEL=...
OPENAI_COMPAT_MODEL_ID=Поиск по документам
RAG_RETRIEVAL_LIMIT=6
RAG_CHAT_HISTORY_LIMIT=8
RAG_FORCE_EXTRACTIVE=false
```

`.env` в GitHub не загружать.

## 13. Как лучше хранить это в GitHub

Лучший вариант:

```text
README.md
.env.example
pyproject.toml
uv.lock
docker-compose.yml
scripts/
src/
tests/
branding/logo.PNG
docs/.gitkeep
```

Не загружать:

```text
.env
.venv
__pycache__
*.pyc
.DS_Store
docs/*.docx
artifacts/
*.log
```

Если локальная папка уже связана с GitHub:

```bash
git status
```

```bash
git add README.md .env.example pyproject.toml uv.lock docker-compose.yml scripts src tests branding docs/.gitkeep .gitignore
```

```bash
git commit -m "Add deployment runbook"
```

```bash
git push
```

Если GitHub ведется через сайт, загружать только файлы из списка выше и не выбирать `.DS_Store`, `.env`, документы и `artifacts`.

## 14. Минимальный чек-лист перед показом

```bash
sudo docker ps
```

```bash
curl http://127.0.0.1:8000/v1/models
```

```bash
curl http://127.0.0.1:3000
```

В браузере:

```text
http://172.19.225.124:3000
```

Проверочный вопрос:

```text
Как организовано рабочее место пользователя?
```

Уточняющий вопрос:

```text
А кто отвечает за это?
```

Ожидается:

```text
ответ на русском
ссылки на источники
модель выбрана в Open WebUI
история чата слева сохраняется
```

## 15. Подготовка DNS, HTTPS и LDAP

Эти файлы можно загрузить в GitHub заранее:

```text
deploy/nginx/glavstroy-llm.conf.example
deploy/env/openwebui.env.example
deploy/env/openwebui.ldap.env.example
deploy/docker/run-openwebui.sh.example
deploy/certs/.gitkeep
```

Реальные сертификаты, пароли и `.env` в GitHub не загружать.

### Nginx

Для текущего HTTP-доступа на порту `80` и отключения браузерного кэша измененных файлов интерфейса:

```bash
cd ~/openwebuig
sudo bash scripts/install_nginx_http_proxy.sh
```

Скрипт полностью проксирует интерфейс на `127.0.0.1:3000`, отключает кэш и перезагружает nginx.

Шаблон:

```text
deploy/nginx/glavstroy-llm.conf.example
```

Когда админы дадут DNS-имя и сертификат:

```bash
sudo cp deploy/nginx/glavstroy-llm.conf.example /etc/nginx/sites-available/glavstroy-llm.conf
```

В файле заменить:

```text
glavstroy-llm.example.local
```

на реальный FQDN.

Также заменить пути:

```text
/etc/nginx/ssl/glavstroy-llm/fullchain.pem
/etc/nginx/ssl/glavstroy-llm/privkey.pem
```

Проверка nginx:

```bash
sudo nginx -t
```

Применить:

```bash
sudo ln -s /etc/nginx/sites-available/glavstroy-llm.conf /etc/nginx/sites-enabled/glavstroy-llm.conf
```

```bash
sudo systemctl reload nginx
```

После nginx пользователям давать HTTPS-ссылку, а не порт `3000`.

### Open WebUI env

Базовый шаблон:

```text
deploy/env/openwebui.env.example
```

На сервере сделать рабочий файл:

```bash
cp deploy/env/openwebui.env.example deploy/env/openwebui.env
```

Заменить:

```text
WEBUI_URL=https://glavstroy-llm.example.local
```

на реальный адрес.

### LDAP

LDAP-шаблон:

```text
deploy/env/openwebui.ldap.env.example
```

Когда дадут данные AD, значения из LDAP-шаблона добавить в:

```text
deploy/env/openwebui.env
```

Главные поля для замены:

```text
LDAP_SERVER_HOST
LDAP_SERVER_PORT
LDAP_CA_CERT_FILE
LDAP_APP_DN
LDAP_APP_PASSWORD
LDAP_SEARCH_BASE
LDAP_SEARCH_FILTER
```

В `LDAP_SEARCH_FILTER` заменить DN группы доступа:

```text
CN=GlavstroyLLM Users,OU=Groups,DC=example,DC=local
```

CA-сертификат положить на сервер в:

```text
deploy/certs/company-ca.crt
```

### Пересоздание Open WebUI с env-файлом

Volume `open-webui` не удалять.

```bash
cd ~/openwebuig
```

```bash
bash deploy/docker/run-openwebui.sh.example
```

Проверить:

```bash
sudo docker ps
```

```bash
curl http://127.0.0.1:3000
```

Если Open WebUI уже был настроен через интерфейс, часть настроек может храниться в базе Open WebUI и иметь приоритет над env. В таком случае менять connection/LDAP через Admin Panel или отдельно отключать persistent config только после backup.
