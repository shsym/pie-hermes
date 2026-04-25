#!/usr/bin/env bash
# Deploy hermes-openai-compat inferlet to a running Pie-on-RunPod pod.
#
# Given a pod SSH address (HOST:PORT), this script:
#   1. scp's setup-engine.sh (from pieclaw), our Pie.toml, our wasm to /tmp
#   2. Runs setup-engine.sh on the pod to bring up pie-vllm + engine
#   3. Kills the standalone engine, waits for GPU memory to drop
#   4. Starts `pie http` with the hermes wasm on the inferlet port
#   5. Sets up an SSH tunnel localhost:<local-port> -> pod:<inferlet-port>
#
# This sidesteps pieclaw/scripts/setup-pie-http-pod.sh's hardcoded openai_compat_v2
# paths so our hermes wasm ships instead.  Keeps the pod-facing setup-engine.sh
# from pieclaw (task #28 tracks its upstream evolution).
#
# Usage:
#   scripts/deploy-hermes-inferlet.sh HOST:PORT
#   scripts/deploy-hermes-inferlet.sh HOST:PORT --model MODEL --max-tokens N
#   scripts/deploy-hermes-inferlet.sh HOST:PORT -- --block-cache-budget-pages=128
#
# Anything after the `--` sentinel is forwarded as launch arguments to the
# inferlet (after pie http's own arg parser).  Use this for inferlet-side
# flags such as block-cache budget, debug toggles, etc.
#
# Env:
#   PIECLAW_DIR        — path to pieclaw checkout (for setup-engine.sh). Default: ../pieclaw
#   WASM_FILE          — override path to the built inferlet wasm.
#   INFERLET_PORT      — pod-side port pie http binds (default 8092).
#   TUNNEL_LOCAL_PORT  — local port for ssh -L (default 9090).
#
# Prereqs: inferlet wasm must be built via ./scripts/build-inferlet.sh first.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIECLAW_DIR="${PIECLAW_DIR:-$REPO_ROOT/../../../pieclaw}"
if [ ! -d "$PIECLAW_DIR" ]; then
    echo "error: PIECLAW_DIR='$PIECLAW_DIR' is not a directory." >&2
    echo "       Set PIECLAW_DIR=/path/to/pieclaw checkout (contains scripts/setup-engine.sh)." >&2
    exit 2
fi

MODEL="Qwen/Qwen2.5-72B-Instruct-AWQ"
MAX_TOKENS="32768"
INFERLET_PORT="${INFERLET_PORT:-8092}"
TUNNEL_LOCAL_PORT="${TUNNEL_LOCAL_PORT:-9090}"
WASM_FILE="${WASM_FILE:-$REPO_ROOT/inferlets/hermes-openai-compat/target/wasm32-wasip2/release/hermes_openai_compat.wasm}"
PIE_TOML="$REPO_ROOT/inferlets/hermes-openai-compat/Pie.toml"
SETUP_ENGINE="$PIECLAW_DIR/scripts/setup-engine.sh"

HOST_PORT=""
INFERLET_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --local-port) TUNNEL_LOCAL_PORT="$2"; shift 2 ;;
        --) shift; INFERLET_ARGS=("$@"); break ;;
        *)
            if [ -z "$HOST_PORT" ]; then HOST_PORT="$1"; shift
            else echo "unknown arg: $1" >&2; exit 2; fi
            ;;
    esac
done

if [ -z "$HOST_PORT" ]; then
    echo "usage: $0 HOST:PORT [--model MODEL] [--max-tokens N] [--local-port N] [-- INFERLET_ARGS...]" >&2
    exit 2
fi

HOST="${HOST_PORT%:*}"
PORT="${HOST_PORT##*:}"

for f in "$WASM_FILE" "$PIE_TOML" "$SETUP_ENGINE"; do
    [ -f "$f" ] || { echo "missing: $f" >&2; exit 2; }
done

# ServerAliveInterval/CountMax tolerates ~60s of intermittent unresponsiveness
# from the pod side (observed during high-CPU windows like model load and
# process teardown) before the SSH session is declared dead.  Without these
# the script would exit non-zero (set -e) on a single transient reset.
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4"
SSH="ssh $SSH_OPTS -p $PORT root@$HOST"
SCP="scp $SSH_OPTS -P $PORT"

echo "=== Step 1: Upload files to pod ==="
$SCP "$SETUP_ENGINE"        "root@$HOST:/tmp/setup-engine.sh"
$SCP "$PIE_TOML"            "root@$HOST:/tmp/Pie.toml"
$SCP "$WASM_FILE"           "root@$HOST:/tmp/hermes_openai_compat.wasm"
$SSH "chmod +x /tmp/setup-engine.sh && ls -la /tmp/setup-engine.sh /tmp/Pie.toml /tmp/hermes_openai_compat.wasm"

echo ""
echo "=== Step 2: Run setup-engine.sh (downloads model, starts engine) ==="
$SSH "bash /tmp/setup-engine.sh --model '$MODEL' --max-batch-tokens $MAX_TOKENS 2>&1 | tail -40"

echo ""
echo "=== Step 3: Fix max_model_len in config.toml ==="
$SSH "sed -i 's/^max_model_len = .*/max_model_len = $MAX_TOKENS/' /root/.pie/engines/eng_benchmark/config.toml"
$SSH "grep -E '^max_(model_len|batch_tokens)' /root/.pie/engines/eng_benchmark/config.toml || true"

echo ""
echo "=== Step 4: Kill standalone engine, wait for GPU idle ==="
# Kill the standalone pie serve (setup-engine.sh left it running); pie http
# starts its own engine from config.toml.
#
# The kill + GPU-memory-poll loop runs in a SINGLE pod-side ssh session.
# Earlier this script ran one ssh per polling iteration; intermittent
# `Connection reset by peer` from the pod's sshd during process teardown
# would propagate through `set -e` and abort the script before the GPU
# returned to idle (observed 2026-04-25 on RunPod A100-SXM4-80GB).
$SSH 'set -u
    pkill -f "pie serve" 2>/dev/null || true
    pkill -f "pie http"  2>/dev/null || true
    for i in $(seq 1 12); do
        sleep 5
        MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d "[:space:]")
        case "$MEM" in
            ""|*[!0-9]*) echo "  iter=$i: GPU mem=unknown (driver busy)"; continue ;;
        esac
        echo "  iter=$i: GPU mem=${MEM} MiB"
        if [ "$MEM" -lt 500 ]; then
            echo "  GPU idle."
            exit 0
        fi
    done
    # Soft fail — Step 6 will catch a stuck pie http via the health-check
    # timeout. Better to proceed and report the real symptom there than
    # abort with a generic "GPU not idle" message.
    echo "  WARN: GPU did not return to idle within 60s; continuing anyway." >&2
'

echo ""
echo "=== Step 5: Start pie http with hermes_openai_compat.wasm ==="
# pie http takes --path/--port/--config; no --manifest.  It bundles the engine,
# so a standalone `pie serve` must not be running simultaneously.
#
# Anything passed after `--` on this script's command line is appended after
# pie http's own argv, which makes it visible to the inferlet via
# `inferlet::get_arguments()`.  Each forwarded arg is shell-escaped via
# `printf %q` so values with whitespace or shell metacharacters survive the
# nested local→ssh→remote-shell parsing.
INFERLET_ARGS_CMD=""
if [ ${#INFERLET_ARGS[@]} -gt 0 ]; then
    INFERLET_ARGS_CMD="-- $(printf '%q ' "${INFERLET_ARGS[@]}")"
    echo "  Forwarding to inferlet: ${INFERLET_ARGS[*]}"
fi
$SSH ": > /tmp/pie-http.log; nohup /opt/pie/pie-vllm-src/pie/pie/.venv/bin/pie http \
        --path /tmp/hermes_openai_compat.wasm \
        --port $INFERLET_PORT \
        --config /root/.pie/engines/eng_benchmark/config.toml \
        $INFERLET_ARGS_CMD \
        >/tmp/pie-http.log 2>&1 &"
echo "  Started pie http in background (log: /tmp/pie-http.log on pod)."

echo ""
echo "=== Step 6: Wait for pie http health ==="
for i in $(seq 1 40); do
    # Health path may not exist; try /v1/models as a liveness signal.
    OK=$($SSH "curl -sfm 3 http://127.0.0.1:$INFERLET_PORT/v1/models 2>/dev/null | head -c 60" || true)
    if [ -n "$OK" ]; then
        echo "  pie http ready (models endpoint live)."
        break
    fi
    if [ "$i" = "40" ]; then
        echo "  pie http did not become ready within 200s." >&2
        $SSH "tail -60 /tmp/pie-http.log" || true
        exit 3
    fi
    sleep 5
done

echo ""
echo "=== Step 7: Set up SSH tunnel (localhost:$TUNNEL_LOCAL_PORT -> pod:$INFERLET_PORT) ==="
pkill -f "ssh.*-L.*$TUNNEL_LOCAL_PORT:127.0.0.1:$INFERLET_PORT.*root@$HOST" 2>/dev/null || true
sleep 1
ssh $SSH_OPTS -f -N -L $TUNNEL_LOCAL_PORT:127.0.0.1:$INFERLET_PORT -p $PORT root@$HOST
sleep 2
OK=$(curl -sfm 5 http://127.0.0.1:$TUNNEL_LOCAL_PORT/v1/models | head -c 60 || true)
if [ -n "$OK" ]; then
    echo "  Tunnel live: http://127.0.0.1:$TUNNEL_LOCAL_PORT"
else
    echo "  Tunnel failed." >&2
    exit 4
fi

echo ""
echo "=== Done ==="
echo "  Pod inferlet: http://127.0.0.1:$INFERLET_PORT (pod)"
echo "  Local tunnel: http://127.0.0.1:$TUNNEL_LOCAL_PORT (this host)"
echo ""
echo "Next steps:"
echo "  python3 scripts/export-variant-kv.py --base-url http://127.0.0.1:$TUNNEL_LOCAL_PORT/v1"
echo "  curl -sN http://127.0.0.1:$TUNNEL_LOCAL_PORT/v1/models"
