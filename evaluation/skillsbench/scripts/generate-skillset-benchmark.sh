#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SKILLSET_NAME="${1:-skills_200}"
OUTPUT_NAME="${2:-generated_${SKILLSET_NAME}}"

python3 "$ROOT/graphskills_benchmark.py" \
  --skillset-name "$SKILLSET_NAME" \
  --output-root "$ROOT/$OUTPUT_NAME"

echo "generated benchmark at $ROOT/$OUTPUT_NAME using $SKILLSET_NAME"
