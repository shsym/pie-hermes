# pie-hermes

Out-of-tree benchmarking harness for running the
[Hermes Agent](https://github.com/shsym/hermes-agent) against a pie
(`openai-compat-v2`) backend and capturing full `/v1/chat/completions`
traffic for prompt-composition analysis.

## Why a separate repo

The hermes-agent fork at `shsym/hermes-agent` is kept pristine so it can
track upstream cleanly. All measurement infrastructure — SDK-level
capture, scenario driver, analyzer, pie configs — lives here and uses
the fork as a git submodule at `hermes-agent/`.

The capture hook is installed without modifying the Hermes tree, by
wrapping `openai.OpenAI.__init__` at interpreter startup (see
`bootstrap/sitecustomize.py` → `bench/install.py`).

## Layout

```
bench/              # driver, analyzer, capture_hook, pie_health_probe, install shim
bootstrap/          # sitecustomize.py — auto-activates capture in subprocess-launched python
configs/            # cli-config.pie.yaml (template; .local.yaml is gitignored)
scripts/            # run_in_container.sh — docker runner
tests/              # unit tests for capture_hook
captures/runs/      # per-campaign run dirs (manifest + scenario capture.jsonl)
hermes-agent/       # submodule → shsym/hermes-agent (main)
```

## Quick start

```bash
# 1. Clone with submodule
git clone --recurse-submodules <url> pie-hermes && cd pie-hermes

# 2. Install deps (optional: only needed for running tests / analyze.py on host)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 3. Set up local config with the live pod URL
cp configs/cli-config.pie.yaml configs/cli-config.pie.local.yaml
# edit base_url in the .local.yaml

# 4. Drive a run (needs a running pie endpoint + hermes-task22 docker image)
BACKEND_TAG=pie-qwen2.5-72b-awq \
ROUND_TAG=round3-fix \
BACKEND_POD_ID=<pod> \
HOST_BASE_URL=http://host.docker.internal:18095/v1 \
  ./scripts/run_in_container.sh

# 5. Analyze
python bench/analyze.py captures/runs/<RUN_ID>
```

## Activation model

Capture is gated on the environment: setting `HERMES_CAPTURE_FILE` (per
scenario) or `HERMES_CAPTURE_DIR` (ad-hoc) activates the hook.
`bench/driver.py` sets `HERMES_CAPTURE_FILE` per scenario, so each
`hermes chat -q` subprocess writes its own `capture.jsonl`. With both
unset, `bench.install` is a no-op.

## Historical reference

`captures/runs/2026-04-21T17-42-49_pie-qwen2.5-72b-awq_round2-deep/` is
the round-2 deep-autonomy capture from task #22 — 7 scenarios,
pod `fs9ajr91180y4r`, inferlet `openai-compat-v2-post@0.2.5`, all green.
Peak prompt 13.4k tok, median per-turn Δ +127 tok. See `analysis.txt`.
