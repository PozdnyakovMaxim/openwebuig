#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source_file="deploy/nginx/glavstroy-llm-http.conf"
available_file="/etc/nginx/sites-available/glavstroy-llm.conf"
enabled_file="/etc/nginx/sites-enabled/glavstroy-llm.conf"

if [[ ! -f "$source_file" ]]; then
  echo "nginx_config_not_found=$source_file"
  exit 1
fi

install -m 0644 "$source_file" "$available_file"
rm -f /etc/nginx/sites-enabled/default
ln -sfn "$available_file" "$enabled_file"

nginx -t
systemctl reload nginx

ready=false
for _ in $(seq 1 30); do
  if curl -fsS http://127.0.0.1/ >/dev/null 2>&1; then
    ready=true
    break
  fi
  sleep 2
done

if [[ "$ready" != "true" ]]; then
  echo "nginx_upstream_not_ready=true"
  exit 1
fi

echo "nginx_proxy_installed=true"
echo "nginx_cache_disabled=true"
