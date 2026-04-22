#!/usr/bin/env bash
# Drive Hermes against the pie pod from inside the hermes Docker container.
#
# Outer-repo edition: the pie-hermes working tree (bench/, bootstrap/,
# configs/, captures/) is mounted into the container; no changes are
# made to the hermes-agent submodule or to the baked hermes image.
# PYTHONPATH is set so that bootstrap/sitecustomize.py runs at every
# python interpreter start (including the one spawned by `hermes
# chat`), which calls bench.install.install() → wraps the SDK's
# openai.OpenAI.__init__ → every client installs the capture tee.
#
# Inputs (env):
#   HOST_BASE_URL      — OpenAI-compat endpoint reachable from inside
#                        the container (default host.docker.internal:18095/v1).
#   BACKEND_TAG        — short slug embedded in the run directory name
#                        (e.g. pie-qwen2.5-72b-awq).
#   ROUND_TAG          — short slug for the round (e.g. round2-deep).
#   BACKEND_POD_ID     — runpod pod id (optional, recorded in manifest).
#   BACKEND_INFERLET   — inferlet name@version (optional, recorded).
#   BACKEND_PROVIDER   — e.g. "pie+openai-compat-v2".
#   BACKEND_MODEL      — e.g. "Qwen/Qwen2.5-72B-Instruct-AWQ".
#   SCENARIO_FILTER    — space-separated slug list to restrict scenarios.
#   HERMES_IMAGE       — Docker image tag for the hermes runtime
#                        (default: hermes-task22).
#   HERMES_CONFIG_PATH — Source cli-config yaml (default: configs/cli-config.pie.yaml).
#                        Override for Phase-1 variants (GPT-branded, Gemini-branded).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

HOST_BASE_URL="${HOST_BASE_URL:-http://host.docker.internal:18095/v1}"
BACKEND_TAG="${BACKEND_TAG:-unknown-backend}"
ROUND_TAG="${ROUND_TAG:-round-unknown}"
HERMES_IMAGE="${HERMES_IMAGE:-hermes-task22}"
HERMES_CONFIG_PATH="${HERMES_CONFIG_PATH:-configs/cli-config.pie.yaml}"

if [[ ! -f "$HERMES_CONFIG_PATH" ]]; then
    echo "error: HERMES_CONFIG_PATH=$HERMES_CONFIG_PATH does not exist" >&2
    exit 2
fi

mkdir -p captures/runs bench/_tmp

# Bake a container-local cli-config that points at the pie endpoint.
# Write into the repo tree (not /tmp) so Docker Desktop's macOS mount
# bridge bind-mounts it as a file, not an autocreated directory.
TMP_CONFIG="$REPO_ROOT/bench/_tmp/pie-run-config.yaml"
trap 'rm -f "$TMP_CONFIG"' EXIT
sed "s|http://REPLACE_ME:8080/v1|${HOST_BASE_URL}|" "${HERMES_CONFIG_PATH:-configs/cli-config.pie.yaml}" > "$TMP_CONFIG"
chmod 644 "$TMP_CONFIG"

docker run --rm -i \
    --name hermes-task22-run \
    -e OPENAI_API_KEY=sk-pie-local-nokey \
    -e HERMES_INFERENCE_PROVIDER=custom \
    -e HOST_BASE_URL="${HOST_BASE_URL}" \
    -e BACKEND_TAG="${BACKEND_TAG}" \
    -e ROUND_TAG="${ROUND_TAG}" \
    -e BACKEND_PROVIDER="${BACKEND_PROVIDER:-pie+openai-compat-v2}" \
    -e BACKEND_MODEL="${BACKEND_MODEL:-Qwen/Qwen2.5-72B-Instruct-AWQ}" \
    -e BACKEND_POD_ID="${BACKEND_POD_ID:-}" \
    -e BACKEND_INFERLET="${BACKEND_INFERLET:-openai-compat-v2@0.2.5}" \
    -e SCENARIO_FILTER="${SCENARIO_FILTER:-}" \
    -e CAPTURES_ROOT=/opt/pie-hermes/captures \
    -e PYTHONPATH=/opt/pie-hermes:/opt/pie-hermes/bootstrap \
    -v "$REPO_ROOT/bench:/opt/pie-hermes/bench:ro" \
    -v "$REPO_ROOT/bootstrap:/opt/pie-hermes/bootstrap:ro" \
    -v "$REPO_ROOT/captures:/opt/pie-hermes/captures" \
    -v "$TMP_CONFIG:/opt/hermes-config/config.yaml:ro" \
    -v "$REPO_ROOT/hermes-agent/run_agent.py:/opt/hermes/run_agent.py:ro" \
    -v "$REPO_ROOT/hermes-agent/agent/prompt_optimizer.py:/opt/hermes/agent/prompt_optimizer.py:ro" \
    --add-host=host.docker.internal:host-gateway \
    --entrypoint /bin/bash \
    "$HERMES_IMAGE" -c '
set -euo pipefail
cp /opt/hermes-config/config.yaml "${HERMES_HOME:-/opt/data}/config.yaml"
python3 -c "import urllib.request; urllib.request.urlopen(\"${HOST_BASE_URL%/v1}/v1/models\", timeout=5); print(\"pie endpoint reachable\")" || {
    echo "Pie endpoint not reachable from container: ${HOST_BASE_URL}" >&2
    exit 1
}
source /opt/hermes/.venv/bin/activate
python3 /opt/pie-hermes/bench/driver.py ${SCENARIO_FILTER}
'
