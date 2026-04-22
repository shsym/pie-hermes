#!/usr/bin/env bash
# Phase-1.5 debug helper: isolate which layer chokes on X-Hermes-Ephemeral.
#
# Sends a sequence of chat/completion requests against the local pie tunnel,
# increasing the X-Hermes-Ephemeral base64 payload size each step.  If one
# tier hangs, you'll see the last "pass" and can diff against the stack layer
# that follows it.
#
# Usage:
#   scripts/debug-idea-g.sh                       # all steps
#   scripts/debug-idea-g.sh --base-url URL        # override tunnel URL
#   scripts/debug-idea-g.sh --step N              # run only step N
#   scripts/debug-idea-g.sh --pod-host HOST:SSH   # also test direct pod URL

set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:9090/v1}"
POD_HOST_PORT=""
ONE_STEP=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --base-url) BASE_URL="$2"; shift 2 ;;
        --pod-host) POD_HOST_PORT="$2"; shift 2 ;;
        --step) ONE_STEP="$2"; shift 2 ;;
        *) echo "unknown: $1" >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

b64_of_len() {
    # Produce ~N bytes of deterministic UTF-8 text, then base64-encode.
    local n="$1"
    python3 -c "
import base64
payload = ('chat_id: 12345\nuser: alice\nplatform: telegram\n' * $n)[:$n]
print(base64.b64encode(payload.encode('utf-8')).decode('ascii'))
"
}

base_payload='{
  "model": "Qwen/Qwen2.5-72B-Instruct-AWQ",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Reply with the single word: OK"}
  ],
  "max_tokens": 8,
  "stream": false
}'

run_step() {
    local n="$1"
    local label="$2"
    local header_args=("$@")
    header_args=("${header_args[@]:2}")

    echo ""
    echo "──── Step $n: $label ────"
    local t0=$(date +%s)
    local out
    local http_code
    out=$(curl -sS -m 120 -o /tmp/idea-g-resp.txt -w '%{http_code}' \
          -H 'Content-Type: application/json' \
          "${header_args[@]}" \
          -X POST "$BASE_URL/chat/completions" \
          -d "$base_payload" 2>&1) || http_code="CURL_ERR"
    http_code="${http_code:-$out}"
    local t1=$(date +%s)
    local dur=$((t1 - t0))
    echo "  elapsed: ${dur}s  http: $http_code"
    if [ "$http_code" = "200" ]; then
        python3 -c "import json,sys; d=json.load(open('/tmp/idea-g-resp.txt')); print('  content:', d.get('choices',[{}])[0].get('message',{}).get('content','?'))" 2>/dev/null || head -c 200 /tmp/idea-g-resp.txt
    else
        echo "  body-head:"
        head -c 400 /tmp/idea-g-resp.txt || true
        echo ""
    fi
}

echo "========================================================"
echo "Idea G hang debug — base=$BASE_URL"
echo "========================================================"

if [ -z "$ONE_STEP" ] || [ "$ONE_STEP" = "1" ]; then
    # Step 1: baseline, no Hermes headers
    run_step 1 "baseline (no X-Hermes headers)"
fi

if [ -z "$ONE_STEP" ] || [ "$ONE_STEP" = "2" ]; then
    # Step 2: empty ephemeral header (should trigger parser's None branch)
    run_step 2 "empty X-Hermes-Ephemeral" \
        -H "X-Hermes-Ephemeral:"
fi

if [ -z "$ONE_STEP" ] || [ "$ONE_STEP" = "3" ]; then
    # Step 3: tiny ephemeral (32 bytes → ~44 chars base64)
    B64=$(b64_of_len 32)
    run_step 3 "tiny ephemeral ($(echo -n "$B64" | wc -c) b64 chars)" \
        -H "X-Hermes-Ephemeral: $B64"
fi

if [ -z "$ONE_STEP" ] || [ "$ONE_STEP" = "4" ]; then
    # Step 4: medium ephemeral (512 bytes → ~684 chars base64)
    B64=$(b64_of_len 512)
    run_step 4 "medium ephemeral ($(echo -n "$B64" | wc -c) b64 chars)" \
        -H "X-Hermes-Ephemeral: $B64"
fi

if [ -z "$ONE_STEP" ] || [ "$ONE_STEP" = "5" ]; then
    # Step 5: fixture size (~1800 bytes → ~2400 chars base64)
    B64=$(b64_of_len 1800)
    run_step 5 "fixture-sized ephemeral ($(echo -n "$B64" | wc -c) b64 chars)" \
        -H "X-Hermes-Ephemeral: $B64"
fi

if [ -z "$ONE_STEP" ] || [ "$ONE_STEP" = "6" ]; then
    # Step 6: variant header (codex) only — baseline for 1A layer
    run_step 6 "X-Hermes-Variant: codex only" \
        -H "X-Hermes-Variant: codex"
fi

if [ -z "$ONE_STEP" ] || [ "$ONE_STEP" = "7" ]; then
    # Step 7: both variant + fixture-sized ephemeral (gateway scenario shape)
    B64=$(b64_of_len 1800)
    run_step 7 "variant=codex + fixture ephemeral" \
        -H "X-Hermes-Variant: codex" \
        -H "X-Hermes-Ephemeral: $B64"
fi

echo ""
echo "========================================================"
echo "Done.  If a step hung: it is the first one that exceeded"
echo "120s.  Compare the last PASS step to the failing step to"
echo "pinpoint the layer — header size, header name, variant+ephemeral interaction."
echo "========================================================"
