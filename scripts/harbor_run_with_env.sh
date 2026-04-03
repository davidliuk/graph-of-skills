#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  # Load repo .env without overriding variables already set in the caller's shell.
  # This keeps explicit per-run overrides authoritative.
  while IFS= read -r line; do
    line=$(printf '%s' "$line" | sed -E 's/^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*/\1=/')
    [[ -z "$line" ]] && continue
    key=${line%%=*}
    value=${line#*=}

    if [[ -z ${!key+x} ]]; then
      export "$key=$value"
    fi
  done < <(grep -v -E '^\s*#|^\s*$' .env)
fi

exec harbor run "$@"
