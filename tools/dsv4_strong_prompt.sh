#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_COMMIT="6c860b23fbe07c04d90975e171c52c938e49dd14"
REF_DIR="${DSV4_REF_DIR:-build-dsv4-reference}"
MODEL_DIR="${1:-${DSV4_MODEL_DIR:-$HOME/model/DeepSeek-V4-Flash-0731}}"
THREADS="${DSV4_THREADS:-8}"
CONTEXT="${DSV4_CONTEXT:-4096}"
MAX_TOKENS="${DSV4_MAX_TOKENS:-160}"

PROMPT="${DSV4_STRONG_PROMPT:-You are given an array a[1..n] with n <= 200000 and q <= 200000 operations. Each operation is either a point update a[i]=x or a range query [l,r]. For each query, return the maximum non-empty subarray sum inside [l,r] after deleting at most one element from that chosen subarray. Derive an associative segment-tree node and merge rule, prove why the merge is correct including boundary cases, give O((n+q) log n) complexity, and provide implementation-ready C99 pseudocode. Do not use brute force or hand-wave the associativity argument.}"

if [ ! -x "$REF_DIR/bin/dsv4" ]; then
  GATE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/dsv4_reference_gate.sh"
  bash "$GATE" "$REF_DIR"
fi

HEAD=$(git -C "$REF_DIR" rev-parse HEAD)
if [ "$HEAD" != "$UPSTREAM_COMMIT" ]; then
  echo "[dsv4-strong] refusing unpinned reference checkout: $HEAD" >&2
  exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
  echo "[dsv4-strong] model directory not found: $MODEL_DIR" >&2
  echo "[dsv4-strong] download the pinned 0731 checkpoint with:" >&2
  echo "  bash $REF_DIR/scripts/download-dsv4.sh '$MODEL_DIR'" >&2
  exit 2
fi

ARGS=(
  --model "$MODEL_DIR"
  --prompt "$PROMPT"
  --max-tokens "$MAX_TOKENS"
  --context "$CONTEXT"
  --threads "$THREADS"
  --temperature 0
  --no-prompt-lookup
)

if [ -n "${DSV4_MEMORY_GIB:-}" ]; then
  ARGS+=(--memory-gib "$DSV4_MEMORY_GIB")
fi

export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-PASSIVE}"

echo "[dsv4-strong] target-only scalar decode; prompt lookup disabled" >&2
echo "[dsv4-strong] model=$MODEL_DIR threads=$THREADS context=$CONTEXT max_tokens=$MAX_TOKENS memory=${DSV4_MEMORY_GIB:-auto}" >&2
exec "$REF_DIR/bin/dsv4" "${ARGS[@]}"
