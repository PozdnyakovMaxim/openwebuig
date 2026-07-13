#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source_file="deploy/nginx/glavstroy-llm-http.conf"
available_file="/etc/nginx/sites-available/glavstroy-llm.conf"
enabled_file="/etc/nginx/sites-enabled/glavstroy-llm.conf"

if [[ ! -f "$source_file" ]]; then
  mkdir -p "$(dirname "$source_file")"
  tee "$source_file" >/dev/null <<'NGINX'
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name _;

    client_max_body_size 25m;

    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
        proxy_buffering off;
        proxy_no_cache 1;
        proxy_cache_bypass 1;
        proxy_hide_header Cache-Control;
        proxy_hide_header Expires;
        add_header Cache-Control "no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0" always;
        add_header Pragma "no-cache" always;
        add_header Expires "0" always;
    }
}
NGINX
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
