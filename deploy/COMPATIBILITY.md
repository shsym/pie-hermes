# Model compatibility matrix

Qualification runs per `bench/scenarios/tool_reliability_subset.md`, on a
single RunPod A40 (48 GB, sm_86) via the `pie-vllm-engine:pieclaw-home`
image. vLLM 0.17.0, torch 2.10.0+cu128. Captured 2026-04-23 for
task #35 (tool-call reliability qualification).

Metrics (per scenario, aggregated):
1. **syntax_valid** — `function.arguments` parses as JSON.
2. **selection_correct** — turn-1 first tool matches the unambiguous expected tool.
3. **completion** — final turn `finish_reason == "stop"` (not stuck in tool loop).
4. **no_tool_discipline** — `shallow-hello` finishes without emitting tool calls.

## Capable tier (48 GB+ Ampere GPU required)

| Model | Weights | KV @ 64K | syntax | selection | completion | discipline | Pass capable bar? | Notes |
|---|---:|---:|---|---|---|---|---|---|
| `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` | 21.06 GiB | 17.0 GiB | 100% (6/6) | 100% (6/6) | 100% (7/7) | 100% (1/1) | **✅ YES** | Runs with default cudagraphs. No special flags. Deep-git-audit used 15/15 max turns but completed cleanly. |
| `Qwen/Qwen3.5-35B-A3B-FP8` (+ `--enforce-eager`) | 34.71 GiB | 3.35 GiB | 100% (6/6) | 100% (6/6) | 100% (7/7) | 100% (1/1) | **✅ YES** | Requires `--enforce-eager` on A40 (see memory `qwen35_fp8_cudagraph_gotcha.md`). ~2-3× slower per token than Int4 due to eager-mode decode. |
| `Qwen/Qwen3.5-35B-A3B-FP8` (default) | 34.71 GiB | 3.35 GiB | — | — | — | — | **❌ NO** | Crashes during FULL cudagraph capture with `AssertionError` in `causal_conv1d_update` (KV-budget-driven). Use the `--enforce-eager` variant above. |

## Working tier (any modern GPU with ≥24 GB recommended)

| Model | Weights | KV @ 64K | syntax | selection | completion | discipline | Pass working bar? | Notes |
|---|---:|---:|---|---|---|---|---|---|
| `Qwen/Qwen3.5-9B` | 17.66 GiB | (not recorded, but fits A40 easily) | 100% (6/6) | 100% (6/6) | 86% (6/7) | 100% (1/1) | **✅ YES** | Passes working-tier bar (≥95/4-of-6/5-of-7/1-of-1). Only miss: `deep-multi-read` ran out of turns mid-tool-loop (15/15, finish_reason=tool_calls). Used more calls per scenario than 35B on shallow tasks (5 vs 2 on shallow-file-head). |

## Bars used

- **Capable tier**: syntax ≥99%, selection ≥5/6, completion ≥6/7, discipline 1/1.
- **Working tier**: syntax ≥95%, selection ≥4/6, completion ≥5/7, discipline 1/1.

Discipline (no spurious tool calls on `shallow-hello`) is a hard floor for
both tiers.

## Dropped candidates (from prior investigation)

| Model | Reason |
|---|---|
| `Qwen/Qwen3.5-27B` | No official quantized variant; bf16 weights (~56 GB) do not fit A40. Community quants not vetted. |
| `Qwen/Qwen3.5-35B-A3B` bf16 | Weights (~70 GB) exceed single-GPU A40 budget even without KV. |
| `Qwen/Qwen2.5-72B-Instruct-AWQ` on A40 | Fits only with reduced context; not a clean single-A40 story. Used only as baseline for analyzer validation (100% all metrics, as expected). |

## Operational requirements

Both capable-tier models require these vLLM-serve flags for tool-calling to
work with Hermes's scenario suite:

```bash
vllm serve <MODEL> \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --host 0.0.0.0 --port 8000
```

**Do NOT use `--tool-call-parser hermes`** — observed experimentally that
the Hermes parser treats Qwen3.5's tool-call format as content, strips
it, and returns empty responses. The `qwen3_coder` parser is what Qwen
publishes for the Qwen3.5 family.

## Scenario-level observations

- Qwen3.5 follows `shallow-hello`'s "no tool call" instruction cleanly on
  all three variants (discipline = 1/1 across the board).
- On `deep-git-audit`, the 35B-A3B-Int4 used all 15 max turns (model prefers
  many smaller tool calls over large ones). Still completed; raising
  max_turns gives headroom if needed.
- 9B hit its `deep-multi-read` budget at 15/15 and didn't reach a final
  answer (fr=tool_calls). 35B-A3B models finished the same scenario in
  11-16 turns with fr=stop.

## Recommendation for task #31 packaging

Primary capable default:
```bash
MODEL=Qwen/Qwen3.5-35B-A3B-GPTQ-Int4
VLLM_EXTRA_ARGS=""
```

Working tier for evaluators/smoke tests:
```bash
MODEL=Qwen/Qwen3.5-9B
VLLM_EXTRA_ARGS=""
```

Quality variant (for A6000 48 GB or A100 80 GB users who want near-lossless):
```bash
MODEL=Qwen/Qwen3.5-35B-A3B-FP8
VLLM_EXTRA_ARGS="--enforce-eager"    # required on A40 48 GB; removable on A100 80 GB
```
