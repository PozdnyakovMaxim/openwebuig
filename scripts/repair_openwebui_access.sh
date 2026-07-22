#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

api_key="${OPENAI_COMPAT_API_KEY:-}"
if [[ -z "$api_key" ]]; then
  api_key="$(PYTHONPATH=src python3 -c 'import os; from document_search.settings import load_env_file; load_env_file(); print(os.getenv("OPENAI_COMPAT_API_KEY") or "")')"
fi

if [[ -z "$api_key" ]]; then
  echo "openai_compat_api_key_missing=true"
  exit 1
fi

loader_api_key="${OPENWEBUI_DOCUMENT_LOADER_API_KEY:-}"
if [[ -z "$loader_api_key" ]]; then
  loader_api_key="$(PYTHONPATH=src python3 -c 'import os; from document_search.settings import load_env_file; load_env_file(); print(os.getenv("OPENWEBUI_DOCUMENT_LOADER_API_KEY") or "")')"
fi
if [[ -z "$loader_api_key" ]]; then
  loader_api_key="$api_key"
fi

OPENAI_COMPAT_API_KEY="$api_key" \
OPENAI_COMPAT_API_BASE_URL=http://host.docker.internal:8000/v1 \
EXTERNAL_DOCUMENT_LOADER_URL=http://host.docker.internal:8000 \
EXTERNAL_DOCUMENT_LOADER_API_KEY="$loader_api_key" \
python3 scripts/fix_openwebui_model_access.py \
  --container document-search-webui \
  --model-id document-search-rag \
  --custom-model-id glavstroy-llm \
  --model-name ГлавстройLLM \
  --activate-pending
unset api_key loader_api_key

python3 scripts/apply_openwebui_branding.py \
  --container document-search-webui \
  --logo branding/logo.PNG \
  --brand-name ГлавстройLLM \
  --model-id document-search-rag \
  --default-model-id document-search-rag

docker restart document-search-webui

openwebui_ready=false
for _ in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:3000/ >/dev/null 2>&1; then
    openwebui_ready=true
    break
  fi
  sleep 2
done

if [[ "$openwebui_ready" != "true" ]]; then
  echo "openwebui_start_timeout=true"
  exit 1
fi

curl -fsS http://127.0.0.1:8000/health
echo
echo "openwebui_ready=true"
