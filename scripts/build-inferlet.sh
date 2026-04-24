#!/usr/bin/env bash
# Build the vendored hermes-openai-compat inferlet to wasm32-wasip2.
#
# Usage:
#   scripts/build-inferlet.sh [--debug]
#
# Output:
#   inferlets/hermes-openai-compat/target/wasm32-wasip2/{release,debug}/hermes_openai_compat.wasm
#
# Prereqs:
#   rustup target add wasm32-wasip2
#
# Notes:
#   - Rustc version pin lives in rust-toolchain.toml next to the crate;
#     if that file is absent this script uses the ambient toolchain.
#   - We build with release profile + LTO (configured in Cargo.toml)
#     because pie loads the module cold on every pod spin-up; artifact
#     size is irrelevant compared to hot-path latency.

set -euo pipefail

PROFILE_FLAG="--release"
PROFILE_DIR="release"
if [[ "${1:-}" == "--debug" ]]; then
  PROFILE_FLAG=""
  PROFILE_DIR="debug"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Guard: the block_cache lookup path relies on the `IMPORTED-CTX` branch
# in pie-vllm's `vllm_sequence_tracker.py` overlay (pie-vllm commit
# 9118d46a, 2026-04-11). Without it, cross-session KV imports corrupt
# responses via the sibling-walk path the tracker historically took —
# see block_cache.rs "Historical" section. Refuse to build if the pinned
# pie-vllm submodule does not contain that commit as an ancestor, so a
# future submodule bump that accidentally drops the fix fails loudly at
# build time rather than silently reintroducing cross-request
# contamination in production.
PIE_VLLM="$REPO_ROOT/pie-vllm"
REQUIRED_PIE_VLLM_COMMIT="9118d46a"
if [[ -d "$PIE_VLLM/.git" || -f "$PIE_VLLM/.git" ]]; then
  if ! git -C "$PIE_VLLM" merge-base --is-ancestor "$REQUIRED_PIE_VLLM_COMMIT" HEAD 2>/dev/null; then
    echo "error: pinned pie-vllm submodule does not contain $REQUIRED_PIE_VLLM_COMMIT" >&2
    echo "       ($REQUIRED_PIE_VLLM_COMMIT is the IMPORTED-CTX tracker fix required by block_cache.rs)" >&2
    echo "       bump the pie-vllm submodule or disable block_cache lookup before building." >&2
    exit 4
  fi
else
  echo "warning: pie-vllm submodule not initialized; skipping IMPORTED-CTX ancestor check." >&2
  echo "         run: git submodule update --init --recursive" >&2
fi

cd "$REPO_ROOT/inferlets/hermes-openai-compat"

# Fail fast with a helpful message when the target isn't installed.
if ! rustup target list --installed | grep -q '^wasm32-wasip2$'; then
  echo "error: wasm32-wasip2 target not installed." >&2
  echo "       run: rustup target add wasm32-wasip2" >&2
  exit 2
fi

cargo build --target wasm32-wasip2 $PROFILE_FLAG

wasm="target/wasm32-wasip2/${PROFILE_DIR}/hermes_openai_compat.wasm"
if [[ ! -f "$wasm" ]]; then
  echo "error: expected wasm artifact not found at $wasm" >&2
  exit 3
fi
size=$(stat -f%z "$wasm" 2>/dev/null || stat -c%s "$wasm")
echo "ok: built ${wasm} (${size} bytes)"
