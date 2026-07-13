#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

container="document-search-webui"
rollback_container="document-search-webui-before-access-fix"
logo="branding/logo.PNG"

if ! docker inspect "$container" >/dev/null 2>&1; then
  echo "container_not_found=$container"
  exit 1
fi

if [[ ! -f "$logo" ]]; then
  echo "logo_not_found=$logo"
  exit 1
fi

curl -fsS http://127.0.0.1:8000/v1/models >/dev/null

image="$(docker inspect "$container" --format '{{.Config.Image}}')"

docker rm -f "$rollback_container" >/dev/null 2>&1 || true
docker stop "$container" >/dev/null
docker rename "$container" "$rollback_container"

rollback() {
  docker rm -f "$container" >/dev/null 2>&1 || true
  docker rename "$rollback_container" "$container" >/dev/null 2>&1 || true
  docker start "$container" >/dev/null 2>&1 || true
  echo "rollback_completed=true"
}

trap rollback ERR

docker run -d \
  --name "$container" \
  --restart unless-stopped \
  -p 3000:8080 \
  -v open-webui:/app/backend/data \
  --add-host host.docker.internal:host-gateway \
  -e WEBUI_NAME=ГлавстройLLM \
  -e DEFAULT_LOCALE=ru-RU \
  -e DEFAULT_USER_ROLE=user \
  -e ENABLE_OLLAMA_API=False \
  -e ENABLE_OPENAI_API=True \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
  -e OPENAI_API_KEY=anything \
  -e OPENAI_API_BASE_URLS=http://host.docker.internal:8000/v1 \
  -e OPENAI_API_KEYS=anything \
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
trap - ERR

echo "openwebui_recreated=true"
echo "model_access_bypass=true"
echo "arena_disabled=true"
