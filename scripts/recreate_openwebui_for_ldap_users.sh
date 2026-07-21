#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

container="document-search-webui"
rollback_container="document-search-webui-before-access-fix"
logo="branding/logo.PNG"
api_key="${OPENAI_COMPAT_API_KEY:-}"

if [[ -z "$api_key" ]]; then
  api_key="$(PYTHONPATH=src python3 -c 'import os; from document_search.settings import load_env_file; load_env_file(); print(os.getenv("OPENAI_COMPAT_API_KEY") or "")')"
fi

if [[ -z "$api_key" ]]; then
  echo "openai_compat_api_key_missing=true"
  exit 1
fi

if ! docker inspect "$container" >/dev/null 2>&1; then
  echo "container_not_found=$container"
  exit 1
fi

if [[ ! -f "$logo" ]]; then
  echo "logo_not_found=$logo"
  exit 1
fi

curl -fsS http://127.0.0.1:8000/health >/dev/null

env_snapshot="$(mktemp /tmp/document-search-webui-env.XXXXXX)"
chmod 600 "$env_snapshot"
trap 'rm -f "$env_snapshot"' EXIT
image="$(docker inspect "$container" --format '{{.Config.Image}}')"
docker inspect "$container" --format '{{range .Config.Env}}{{println .}}{{end}}' > "$env_snapshot"

docker rm -f "$rollback_container" >/dev/null 2>&1 || true
docker stop "$container" >/dev/null
docker rename "$container" "$rollback_container"

rollback() {
  rm -f "$env_snapshot"
  docker rm -f "$container" >/dev/null 2>&1 || true
  docker rename "$rollback_container" "$container" >/dev/null 2>&1 || true
  docker start "$container" >/dev/null 2>&1 || true
  echo "rollback_completed=true"
}

trap rollback ERR

export OPENAI_API_KEY="$api_key"
export OPENAI_API_KEYS="$api_key"

docker run -d \
  --name "$container" \
  --restart unless-stopped \
  -p 3000:8080 \
  --volumes-from "$rollback_container" \
  --add-host host.docker.internal:host-gateway \
  --env-file "$env_snapshot" \
  -e WEBUI_NAME=ГлавстройLLM \
  -e DEFAULT_LOCALE=ru-RU \
  -e DEFAULT_USER_ROLE=user \
  -e ENABLE_OLLAMA_API=False \
  -e ENABLE_OPENAI_API=True \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
  -e OPENAI_API_KEY \
  -e OPENAI_API_BASE_URLS=http://host.docker.internal:8000/v1 \
  -e OPENAI_API_KEYS \
  -e DEFAULT_MODELS=document-search-rag \
  -e DEFAULT_PINNED_MODELS=document-search-rag \
  -e BYPASS_MODEL_ACCESS_CONTROL=True \
  -e ENABLE_EVALUATION_ARENA_MODELS=False \
  -e EVALUATION_ARENA_MODELS='[]' \
  -e ENABLE_COMMUNITY_SHARING=False \
  -e OFFLINE_MODE=True \
  -e HF_HUB_OFFLINE=1 \
  -e ENABLE_VERSION_UPDATE_CHECK=False \
  "$image" >/dev/null
unset OPENAI_API_KEY OPENAI_API_KEYS api_key

ready=false
for _ in $(seq 1 120); do
  if curl -fsS http://127.0.0.1:3000/ >/dev/null 2>&1; then
    ready=true
    break
  fi
  sleep 2
done

if [[ "$ready" != "true" ]]; then
  echo "openwebui_start_timeout=true"
  rollback
  trap - ERR
  exit 1
fi

bash scripts/repair_openwebui_access.sh

docker rm -f "$rollback_container" >/dev/null
rm -f "$env_snapshot"
trap - ERR
trap - EXIT

echo "openwebui_recreated=true"
echo "model_access_bypass=true"
echo "arena_disabled=true"
