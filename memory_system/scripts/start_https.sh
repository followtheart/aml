#!/usr/bin/env bash
# Run from any directory; configuration is inherited from the environment.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ${1:-} == --help ]]; then
  cat <<'EOF'
Usage: AML_API_KEY=... SSL_CERTFILE=... SSL_KEYFILE=... bash start_https.sh
Optional: HOST=0.0.0.0 PORT=443 PYTHON=/path/to/python AML_FAKE=1
Existing AML_* model/storage configuration is inherited. No .env is loaded.
EOF
  exit 0
fi
: "${AML_API_KEY:?Set AML_API_KEY for external API authentication}"
: "${SSL_CERTFILE:?Set SSL_CERTFILE to the PEM certificate chain}"
: "${SSL_KEYFILE:?Set SSL_KEYFILE to the PEM private key}"
[[ -r "$SSL_CERTFILE" && -r "$SSL_KEYFILE" ]] || { echo 'TLS certificate/key is not readable' >&2; exit 1; }
if [[ -z ${PYTHON:-} ]]; then
  PYTHON=python3
  [[ ! -x "$ROOT/.venv/bin/python" ]] || PYTHON="$ROOT/.venv/bin/python"
fi
# Keep caller-relative certificate and database paths intact.
exec "$PYTHON" -m uvicorn app.main:app --app-dir "$ROOT" \
  --host "${HOST:-0.0.0.0}" --port "${PORT:-443}" \
  --ssl-certfile "$SSL_CERTFILE" --ssl-keyfile "$SSL_KEYFILE" \
  --no-proxy-headers
