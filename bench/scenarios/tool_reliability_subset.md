# Tool-call reliability scenario subset

The subset used by the `hc task #35` / task-list `T3`/`T4` qualification runs
against Qwen3.5 candidates. Goal: measure base-model tool-use reliability
against the pie+inferlet or vanilla-vllm backend, not anything
substrate-dependent.

## Included (7 scenarios)

Chosen because (a) each exercises real tool-call emission, (b) the expected
first tool is unambiguous from the prompt so `selection_correct` is
well-defined, and (c) none depend on Phase-2 ephemeral/skills substrate.

| Scenario | First-turn expected tool | Max turns | Role in matrix |
|---|---|---:|---|
| `shallow-hello` | *none* — no-tool discipline probe | 3 | Does the model refrain from spurious tool calls? |
| `shallow-file-head` | `read_file` | 5 | Single file read + synthesis |
| `shallow-terminal` | `terminal` | 5 | Single shell call + report |
| `deep-search-read` | `search_files` | 15 | Multi-tool: search → 3× read → synthesis |
| `deep-git-audit` | `terminal` | 15 | Large terminal outputs, multi-turn |
| `deep-multi-read` | `read_file` | 15 | 3× file reads + comparative synthesis |
| `deep-breadth-audit` | `search_files` | 20 | Longest scenario; breadth-first reads |

## Excluded (5 scenarios)

Not about base-model reliability; they measure our Phase-2 substrate work.

| Scenario | Why excluded |
|---|---|
| `gateway-ephemeral-chat` | Needs gateway shim (Idea G) to inject ephemeral metadata. Without it, runs degenerate / produces meaningless comparisons. |
| `gateway-ephemeral-chat-3turn` | Same as above, multi-turn. |
| `probe-ephemeral-handle-recall` | Probe specifically designed to fail without Phase-2 semantic re-injection. Not a reliability signal. |
| `probe-read-context-one-call` | Phase-2.1 read-ctx probe. |
| `probe-skills-handle-recall` | Phase-2.2 skills-index probe. |

## Metrics computed per scenario

From `bench/analyze_tool_reliability.py`:

1. **syntax_valid_rate** — share of emitted tool-call `function.arguments`
   strings that parse as JSON. Fail < 0.99.
2. **selection_correct** — did turn-1's FIRST tool equal the scenario's
   expected tool? Only evaluated for scenarios listed in `EXPECTED_TOOL`
   (everything above except `shallow-hello`).
3. **completion** — did the scenario's final turn finish with
   `finish_reason == "stop"` (i.e. the model produced a user-facing answer,
   not a stuck tool-call loop or truncation).
4. **no_tool_discipline** — only evaluated for `shallow-hello`. Fails if
   turn-1 emitted any tool_calls.

## Pass bar for "capable"

- `syntax_valid_rate` ≥ 0.99 aggregated across all included scenarios.
- `selection_correct` ≥ 5 of 6 (Qwen2.5-72B baseline: 6/6).
- `completion` ≥ 6 of 7.
- `no_tool_discipline` = 1/1 (hard floor).

Below this bar → disqualified from the capable tier; record failure mode
in `deploy/COMPATIBILITY.md`.

## Pass bar for "working" tier (Qwen3.5-9B)

Same metrics, slightly relaxed:
- `syntax_valid_rate` ≥ 0.95
- `selection_correct` ≥ 4 of 6
- `completion` ≥ 5 of 7
- `no_tool_discipline` = 1/1 (same hard floor — a 9B that emits spurious
  tool calls on shallow-hello is not a working tier we ship).

## Driver invocation

`bench/driver.py` reads `SCENARIOS` in full and runs everything by default.
To restrict to this subset, set `SCENARIO_FILTER` env var (existing
`scripts/run_in_container.sh` already pipes this through):

```bash
SCENARIO_FILTER="shallow-hello shallow-file-head shallow-terminal deep-search-read deep-git-audit deep-multi-read deep-breadth-audit" \
  scripts/run_in_container.sh
```

## vLLM-side flags for Qwen3.5 (record for T3/T4 invocations)

- `--reasoning-parser qwen3` — CRITICAL. Without it, Qwen3.5's thinking-mode
  output lands in `response.message.content` and Hermes parses the wrong
  text. With it, vLLM separates `reasoning_content` from `content`; Hermes
  only reads `content`.
- `--max-model-len 65536` — match Hermes's 64K floor (cli-config.pie.yaml).
- `--gpu-memory-utilization 0.90` — default.
- `--enforce-eager` — **required for FP8 on A40** per memory
  `qwen35_fp8_cudagraph_gotcha.md`. Not needed for Int4.
- `--trust-remote-code` — Qwen3.5 config calls custom modelling code.

## Hermes-side config

New `configs/cli-config.qwen35.yaml`:
- `model.provider: custom` (OpenAI-wire dispatch, unchanged)
- `model.base_url: http://REPLACE_ME/v1`
- `model.context_length: 65536` (match vLLM --max-model-len)
- `agent.tool_use_enforcement: true` (Qwen3.5 is still Qwen-family →
  explicit steering still needed, same as 2.5; see pie-config rationale)
- `agent.max_turns: 20` (covers deep-breadth-audit max)
