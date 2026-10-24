#!/usr/bin/env bash
# Standalone TLS service; no files or modules from memory_system are loaded.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ${1:-} == --help ]]; then
  cat <<'EOF'
Usage: MC_API_KEY=... SSL_CERTFILE=... SSL_KEYFILE=... bash start_https.sh
Optional: HOST=0.0.0.0 PORT=443 PYTHON=/path/to/python MC_FAKE=true
All MC_* settings are inherited. No .env is loaded.
EOF
  exit 0
fi
: "${MC_API_KEY:?Set MC_API_KEY for external API authentication}"
: "${SSL_CERTFILE:?Set SSL_CERTFILE to the PEM certificate chain}"
: "${SSL_KEYFILE:?Set SSL_KEYFILE to the PEM private key}"
[[ -r "$SSL_CERTFILE" && -r "$SSL_KEYFILE" ]] || { echo 'TLS certificate/key is not readable' >&2; exit 1; }
if [[ -z ${PYTHON:-} ]]; then
  PYTHON=python3
  [[ ! -x "$ROOT/.venv/bin/python" ]] || PYTHON="$ROOT/.venv/bin/python"
fi
exec "$PYTHON" -m uvicorn memory_connection.app.https_app:app --app-dir "$(dirname -- "$ROOT")" \
  --host "${HOST:-0.0.0.0}" --port "${PORT:-443}" \
  --ssl-certfile "$SSL_CERTFILE" --ssl-keyfile "$SSL_KEYFILE" \
  --no-proxy-headers --no-access-log --timeout-graceful-shutdown 30
