#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_REPO="https://github.com/shyringo/deepseek-v4-flash-0731-in-c.git"
UPSTREAM_COMMIT="6c860b23fbe07c04d90975e171c52c938e49dd14"
REF_DIR="${1:-build-dsv4-reference}"
JOBS="${DSV4_REF_JOBS:-2}"

if [ ! -d "$REF_DIR/.git" ]; then
  rm -rf "$REF_DIR"
  git clone --no-checkout "$UPSTREAM_REPO" "$REF_DIR"
fi

git -C "$REF_DIR" fetch --depth 1 origin "$UPSTREAM_COMMIT"
git -C "$REF_DIR" checkout --detach "$UPSTREAM_COMMIT"

make -C "$REF_DIR" clean
make -C "$REF_DIR" -j"$JOBS"
make -C "$REF_DIR" test
make -C "$REF_DIR" portable

printf 'DSV4_REFERENCE_TARGET_GATE_PASS commit=%s\n' "$UPSTREAM_COMMIT"
