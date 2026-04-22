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
