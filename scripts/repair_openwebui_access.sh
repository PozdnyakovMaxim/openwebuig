#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 scripts/fix_openwebui_model_access.py \
  --container document-search-webui \
  --model-id document-search-rag \
  --custom-model-id glavstroy-llm \
  --model-name ГлавстройLLM \
  --activate-pending

python3 scripts/apply_openwebui_branding.py \
  --container document-search-webui \
  --logo branding/logo.PNG \
  --brand-name ГлавстройLLM \
  --model-id document-search-rag \
  --default-model-id document-search-rag

docker restart document-search-webui

sleep 8

curl -fsS http://127.0.0.1:8000/v1/models
echo
