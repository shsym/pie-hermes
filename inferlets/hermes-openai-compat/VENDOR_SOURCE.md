# Vendor source

This crate was forked from pieclaw/inferlets/openai-compat-v2.

- Upstream branch at fork: `task19-phase-b-instrumentation`
- Upstream HEAD at fork:   `52bce2dfce1719639121e188d05a660df32a3885`
- Fork date:               `2026-04-22`
- Rename:                  `openai-compat-v2` → `hermes-openai-compat` (Task 0.5)

## Sync policy

Periodic merges from pieclaw are owned by **task #28**. Do NOT pull directly
from pieclaw from this crate — route all merges through the
`scripts/sync-pieclaw-inferlet.sh` helper so the SYNC_LOG.md stays truthful.

## Why a rename

The fork is hermes-specific: it will diverge with `skill_view`, `read_context`,
variant-KV injection, and out-of-band metadata handling. Keeping the original
crate name would make `Pie.toml`-level artifact conflicts possible on pods
that load both inferlets.

## Local divergences from upstream

Every change listed here needs to be (a) carried forward on the next sync
from pieclaw, or (b) upstreamed and then removed here. Owner: task #28.

### 1. `#[cfg(target_family = "wasm")]` gate on `#[wstd::http_server]` in `src/lib.rs`

**Added:** 2026-04-22, Task 0.8 (pie-hermes).

**Reason:** pieclaw's `task19-phase-b-instrumentation` branch added
`rlib` crate-type and host-side unit tests but left the
`#[wstd::http_server]` macro unconditional. The macro expands to WASI
component imports (`wasi:http/incoming-handler@0.2.4`) that only the wasm
target can resolve. Without the gate, plain `cargo test` (which also
builds the `cdylib`) fails to link on the host.

**Fix:** gate the macro on `cfg(target_family = "wasm")` so the entrypoint
is only emitted when building for a wasm target. Host `cargo test` (no
`--target` flag → defaults to the host triple) sees the function elided;
release builds target `wasm32-wasip2` and are unaffected.

**Upstream-worthy:** yes. The same bug exists in
pieclaw/inferlets/openai-compat-v2. Task #28 should push this fix back
to pieclaw on its next sync pass; once accepted upstream, remove from
this divergence list.

### 2. `src/variant.rs` + `POST /v1/pie/variant/export` route (Task 1A.2)

**Added:** 2026-04-22, Task 1A.2 (pie-hermes Phase 1).

**Reason:** Idea A optimization pre-exports named KV handles per model-family
variant (codex / google / none) so the X-Hermes-Variant header can trigger
an import at request time instead of paying the ~1.1k–2.5k-token cost of
the GOOGLE_MODEL_OPERATIONAL_GUIDANCE / OPENAI_MODEL_EXECUTION_GUIDANCE
/ TOOL_USE_ENFORCEMENT_GUIDANCE blocks in every request.

**Surface:** one new source module (`src/variant.rs`) and one new route arm
in `src/lib.rs`. The route reads a JSON body {variant_name, text}, prefills
a standalone Context, and calls `ctx.queue().export_resource_sync(
Resource::KvPage, ptrs, "hermes-variant-<name>")`. Consumed by
`scripts/export-variant-kv.py` on the host side.

**SDK note:** `export_resource_sync` lives on `Queue`, not `Model` (the
originating task brief stated Model; the pinned pie-vllm submodule puts
it on Queue at `pie/sdk/rust/inferlet/src/lib.rs:323`). We call it via the
Context's own queue (`ctx.queue()`), matching the pattern used by the
existing `/v1/debug/kv-exports` route.

**Upstream-worthy:** yes, conditionally. Pieclaw may want this route as a
generic "KV pre-population" utility; the specifics of variant-naming are
pie-hermes business. Task #28 can decide at merge time whether to upstream
the route skeleton + make the handle namespace client-configurable.

### 3. `X-Hermes-Variant` header → variant-KV injection (Task 1A.3)

**Added:** 2026-04-22, Task 1A.3 (pie-hermes Phase 1).

**Reason:** Closes the loop from Task 1A.2's export route.  When a
`POST /v1/chat/completions` request arrives with `X-Hermes-Variant: codex`
or `X-Hermes-Variant: google`, the handler short-circuits to a new
`prepare_variant_execution` path that:

1. Loads `VariantMetadata` (token_ids + kv_page_last_len) from the
   persistent store at key `v1.variant.<name>.meta`.
2. Calls `queue.import_kv_pages("hermes-variant-<name>")` to import the
   pre-exported pages.
3. Constructs a Context via `Context::from_imported_state(model, kv_pages,
   token_ids, kv_page_last_len)` — the variant prefix is already in GPU KV
   cache, so those tokens cost zero prefill FLOPs.
4. Appends only the chat-specific tail via `ctx.fill_tokens(chat_tokens)`.

The `VariantMetadata` type and `save_metadata`/`load_metadata` helpers are
in `src/variant.rs` (same file as the export route).  `handle_export` now
calls `save_metadata` after a successful `export_resource_sync`, so export
and import metadata stay in sync.

**Phase-1 scope limitation:** applies ONLY to the non-session,
non-registered-prefix path.  When the request has `pie_session` or
`pie_prompt.mode == "registered_prefix"`, the existing path is used
unchanged.  Coexisting variant + session KV semantics on turn 2+ require
storing the variant tokens in session state and re-matching against them on
subsequent turns — deferred to a later phase.  Session-ID auto-derivation
(from bearer + system prompt) is also skipped when the variant header is
present, to ensure variant requests always land on the non-session path.

**Surface:** `src/variant.rs` (VariantMetadata, save/load_metadata,
parse_variant_header), `src/handler.rs` (prepare_variant_execution,
updated prepare_execution signature, updated handle_chat_completions),
`src/lib.rs` (variant_header extraction + wiring).

**Upstream-worthy:** the generic `from_imported_state` usage pattern and
the `parse_variant_header` helper are reusable; the specific variant
names/store keys are pie-hermes business.  Task #28 decides at merge time.

### 4. `X-Hermes-Ephemeral` header parsing in `src/variant.rs` + `src/lib.rs` + `src/handler.rs` (Task 1G.3)

**Added:** 2026-04-22, Task 1G.3 (pie-hermes Phase 1).

**Reason:** Idea G routes ephemeral metadata (chat_id, user handle, reactions,
platform quirks) out-of-band via `X-Hermes-Ephemeral: <base64-utf8>` header
instead of appending it as prompt text (see pie-hermes `EphemeralOOBRouter`,
task 1G.2). Phase-1 scope on the inferlet side is minimal: validate the
header is well-formed base64 and log receipt (length-only — never decoded
into logs) so Phase-1G plumbing is observable on the live pod. Actual
platform-specific post-generation formatting is deferred to a later phase.

**Surface:** new `parse_ephemeral_header` helper co-located with
`parse_variant_header` in `src/variant.rs` (splitting into a `headers.rs`
module for one additional function is premature; both helpers parse
`X-Hermes-*` request headers and the file already owns that shape). New
header extraction in the chat-completions router arm in `src/lib.rs`. New
optional `ephemeral_header: Option<String>` parameter on
`handler::handle_chat_completions`. New `base64 = "0.22"` Cargo dependency
with `default-features = false, features = ["alloc", "std"]` — kept minimal
for wasm32-wasip2.

**Logging discipline:** `parse_ephemeral_header` intentionally returns the
raw base64 string (not the decoded bytes) so decoded user data never crosses
the validation boundary via return value. The handler no longer logs anything
on the ephemeral code path — see divergence #5.

**Upstream-worthy:** conditionally. Pieclaw may want a generic OOB-metadata
header pattern; task #28 can decide at merge time whether to carve the
helper into a generic `headers.rs` module.

### 5. `eprintln!()` is forbidden in the synchronous prefix of `handle_chat_completions` (Task 1.5-G debug)

**Added:** 2026-04-22, Phase 1.5 Idea G hang debug (pie-hermes).

**Reason:** Calling `eprintln!()` from inside the async
`handle_chat_completions` handler *before the first `.await`* causes every
incoming request that reaches the `eprintln` call to hang indefinitely
under `wstd = 1.0.1` on `wasm32-wasip2`: curl observes HTTP:000 / 0 bytes
received. The hypothesis is that `std::io::stderr()` on `wasm32-wasip2`
maps to a blocking WASI call, and invoking it before the executor has
suspended the task stalls wstd's single-threaded runtime before any
response bytes are written. Verified empirically via
`scripts/debug-idea-g.sh`: removing the single `eprintln!(...)` that fired
when `X-Hermes-Ephemeral` was non-empty turned every hanging step
(steps 3-7, all >120 s) into a 0-1 s 200 response.

**Constraint documented in code:** the ephemeral code path in
`src/handler.rs:175-181` carries a comment explaining the prohibition and
binds the unused `ephemeral_header` with `let _ = &ephemeral_header;` to
keep the type signature meaningful.

**Existing inherited `eprintln!()` calls:** two pre-existing calls
(`src/handler.rs:~430` in the constrained-sampler fallback, `~1965` in
the `export_kv_pages_sync` failure path) fire only in error branches that
run *after* the handler has suspended on `.await` at least once. They
have not surfaced this symptom in production on pieclaw; leave them in
place. Any *new* logging should either (a) happen behind an `.await` or
(b) use a non-blocking async mechanism.

**Upstream-worthy:** yes, as a documentation change. If pieclaw adds new
sync-prefix logging to `handle_chat_completions`, they will hit the same
bug. A proper fix is an async-safe logger; task #28 should mirror this
note upstream and coordinate on the shared solution.

### 6. `decode_ephemeral_payload` + ephemeral re-prefill via `request.messages` mutation (Task 2.0)

**Added:** 2026-04-22, Phase 2.0 (pie-hermes).

**Reason:** Phase 1.5 left ephemeral-header content visible to the inferlet
but invisible to the model (`let _ = &ephemeral_header;` at the top of
`handle_chat_completions`). Phase 2.0 closes the loop by decoding the base64
payload and inserting a synthesized system `ChatMessage` at the END of the
existing system block in `request.messages`, BEFORE `prepare_execution` runs.
The chat template (`format_chat_tokens` → engine-side `format_chat_rpc`)
then renders the new system message with proper role markers.

The injection happens AFTER the cached system prefix region in token-space,
so block-cache prefix matching on the leading system tokens is preserved on
the **direct** (non-session, non-registered, non-variant) path.

**Surface:**
- New `variant::decode_ephemeral_payload(raw_b64: &str) -> Option<String>`
  helper (sibling of `parse_ephemeral_header`).  Returns `None` for empty
  bytes / non-base64 / non-UTF-8 — defense-in-depth even though
  `parse_ephemeral_header` already validates well-formedness.
- New optional `PieCacheTelemetry.ephemeral_tokens_appended: Option<u32>`
  field; only populated on the ephemeral path.  Approximation:
  `tokenizer.tokenize(decoded_text).len()`; under-estimates by ~6-8 tokens
  per request because the chat template's role markers
  (`<|im_start|>system\n...<|im_end|>\n`) are not counted.
- Changed handler at `src/handler.rs:175-223` (decode + inject) and
  `:285-296` (post-prep telemetry mutation).  Bound `prepared` as `mut`.
- Updated 12 `PieCacheTelemetry { ... }` initializer sites in
  `src/handler.rs` to set `ephemeral_tokens_appended: None`.

**Logging discipline preserved:** decoded payload is consumed by
`request.messages` insertion only, never written to a log line.
`parse_ephemeral_header` continues to return the raw base64 for length-only
logging behind `.await`.

**Acknowledged Phase-2.0 tradeoffs (surfaced by code review of Task 1B):**

1. **Variant + ephemeral interaction.**  When `X-Hermes-Variant` and
   `X-Hermes-Ephemeral` are BOTH set, the injected ephemeral system message
   appears in `request.messages` before `prepare_variant_execution` runs.
   That path imports the pre-exported variant KV and then fills
   `format_chat_tokens(request.messages, ...)` (which now includes the
   ephemeral block) on top.  This is COHERENT only because today's variant
   `meta.token_ids` represents the system-prefix block only; the
   chat-template render of `request.messages` yields the ephemeral block
   followed by the user/assistant turns, which lands on top of the
   variant prefix as a clean continuation.  If a future variant export
   ever included user/assistant turns, the ephemeral injection would
   land mid-conversation and degrade response quality.  Constraint
   recorded for the variant-export pipeline: **variant KV must remain
   system-prefix-only** as long as Phase-2.0 ephemeral injection is in
   flight.

2. **Session path: ephemeral content invalidates prefix-cache match on
   turn N+1.**  The session path
   (`prepare_session_execution`) compares the previous turn's
   `incoming_tokens` against this turn's render.  Because the ephemeral
   system message lives high in the system block, any variation in
   ephemeral content between turns shifts tokens early in the stream
   and breaks the prefix match very early — invalidating the entire
   session KV for that turn.  The session falls through to
   `block_cache_hit` or `full_prefill`.  This is an **accepted tradeoff
   for Phase-2.0 shape (a)**: ephemeral content is per-message by
   definition, and the prefix-cache hit on the *block-cache* path
   (which keys on content hash, not session tokens) still recovers the
   leading system region.  Phase-2.0 does NOT optimize the session +
   ephemeral combination; that work is deferred to a later phase that
   would need to pin a per-turn ephemeral-message position the session
   path can normalize over.

**Upstream-worthy:** conditionally — same as #4.  The "OOB metadata
appended after cached prefix via `messages` mutation" pattern generalises
beyond hermes; task #28 decides at merge time.  Both tradeoffs above
should accompany any upstream patch so reviewers know what the design
covers and what it explicitly defers.

### 7. `src/context_section.rs` + `POST /v1/pie/context-section/register` route (Task 2.1, Idea D)

**Added:** 2026-04-22, Phase 2.1 (pie-hermes).

**Reason:** Idea D (context-files summary) replaces verbatim AGENTS.md
(and similar) content in the system prompt with a compact index. The
section bodies are registered server-side as named KV-prefix handles
keyed by `(section_id, body_hash)`. At inference time, a `read_context`
tool intercept (Task 2.2 / divergence #8) imports the prefix and surfaces
the body to the model on demand, instead of paying the body's prompt-
token cost on every request.

**Surface:** one new source module (`src/context_section.rs`); one new
route arm in `src/lib.rs` (`POST /v1/pie/context-section/register`); a
side-table at `v1.section.<section_id>.latest_hash` so the read_context
intercept can resolve a section_id to the latest body without the agent
re-supplying the hash per-call.

**Storage keys (under inferlet::store_set namespace):**
- `v1.section.<section_id>.<body_hash>.meta` — JSON SectionMetadata.
- `v1.section.<section_id>.latest_hash`      — most recently registered hash.
- KV-pages handle: `hermes-section-<section_id>-<body_hash>`.

**Upstream-worthy:** conditionally — same as #2 / #6. The "register a
named prefix and intercept a tool call to import it" pattern is generic;
the specific naming scheme is pie-hermes business. Task #28 decides.

### 8. Tool-result body → registered-handle detection + telemetry (Task 3.0)

**Added:** 2026-04-23, Phase 3.0 (pie-hermes).

**Amended:** 2026-04-23 (same session), Phase 3.0 rescope. The original
framing ("substrate for a deferred mid-stream prefill-skip") was
invalidated during implementation — see "Rescope history" below. The
mechanism is unchanged; what it *measures* has been reframed.

**Scope-limitation upfront:** this divergence ships DETECTION +
TELEMETRY ONLY. The KV path is unchanged. The registered pages from
divergence #7 still sit idle; the normal prefill runs. Phase 3.0 is now
a characterization experiment — measuring the baseline behavior of the
existing caches under synthetic pressure — not groundwork for a new KV
mechanism. See "Rescope history" below.

**Reason:** the telemetry field is the discriminator for the Phase-3.0
APC-pressure benchmark (`bench/drive_pressure.py`, `bench/analyze_
pressure.py`). Per-request it answers: did THIS tool-result body land
on a registered (section_id, body_hash)? Cross-referenced against
`pie_cache.prefill_tokens_skipped` it tells us whether existing
caches (block_cache, session_kv, prefix_checkpoint) already covered
the body's prefill, or whether a body-targeted mechanism would add
savings on top.

Secondary uses preserved from the original framing: verifies (a)
hermes-agent's boot-time registrar reached the inferlet, (b) the
body-hash wire format matches across Python and Rust, (c) the
`read_context` tool path delivers bodies whose hashes line up with
what was registered. Those were preconditions for prefill-skip; they
are now preconditions for interpreting the benchmark's findings.

**Surface:**
- `src/context_section.rs`: new `detect_tool_result_matches(messages,
  model) -> Option<u32>` helper + `body_hash_16(body) -> String`.
  `body_hash_16` produces the same 16-char SHA-256 hex prefix that
  hermes-agent's `ContextSection.body_hash` emits (cross-checked in unit
  tests — a drift here would silently break lookups on byte-identical
  bodies).
- `src/types.rs`: new `PieCacheTelemetry.tool_result_tokens_imported:
  Option<u32>` field, `skip_serializing_if = Option::is_none`. Wire
  shape parity with the existing `ephemeral_tokens_appended`
  approximation.
- `src/handler.rs:~309`: scan-and-populate, right after the Phase-2.0
  ephemeral-telemetry block. 12 existing `PieCacheTelemetry { ... }`
  initializer sites updated to carry `tool_result_tokens_imported: None`.
- No changes to the register route, no storage-format changes.

**Telemetry semantics:**
- `None`: no `role:"tool"` messages in the request.
- `Some(0)`: tool messages present; none matched a registered handle
  (either malformed envelope, unknown section_id, or hash drift).
- `Some(n)`: sum of `tokenizer.tokenize(body).len()` for every matched
  tool message. Reports the body-span footprint only; the tool-role
  markers added by the chat template (`<|im_start|>tool\n...<|im_end|>`)
  are NOT counted, so the actual prompt-token footprint of the tool
  message is ~4–6 tokens larger than the reported value. Using the body
  span rather than `SectionMetadata.token_ids.len()` (which is the
  `fill_system`-wrapped register-time count) gives a more honest
  "would-be-saved" number and keeps the units consistent with
  `ephemeral_tokens_appended`.

**Content-hash only, no server-side dispatch:** the detector does NOT
inspect the preceding assistant message's `tool_calls` or `section_id`
arguments. Match is purely on the `(section_id, body_hash)` pair parsed
out of the tool-result JSON envelope. Server-side tool dispatch is
explicitly deferred (task #32 trigger #3).

**Rescope history (why the original "substrate for prefill-skip"
framing didn't survive).** The plan's prefill-skip path assumed a
future mid-stream KV splice. Two blockers surfaced:

1. **RoPE positional wall.** KV pages registered via divergence #7
   are exported from a context built by `ctx.fill_system(body)`, so
   they are (a) wrapped in `<|im_start|>system\n…<|im_end|>\n` role
   markers at the token level and (b) RoPE-rotated for absolute
   positions `0..N`. A `role:"tool"` span in a real chat render has
   DIFFERENT role markers (`<|im_start|>tool\n…`) and lives at
   absolute position `M..M+N`. Fixing either requires SDK surface
   that doesn't exist: `Context::from_imported_state*` is prefix-only,
   and `from_imported_state_with_positions` explicitly notes
   "flashinfer does not re-apply RoPE … the rewritten value is only
   ever read by `prepare_forward_pass` to derive `last_pos_id + 1`
   for the new tokens." K values baked at positions `0..N` produce
   incorrect attention at positions `M..M+N`.

2. **Existing caches may already cover this.** pie-vllm does NOT
   vendor vanilla vLLM APC — `pie_worker/vllm_runtime.py` comments:
   "KV export/import (prefix_caching): Only attention-layer KV is
   exported" — cross-session reuse is gated by explicit
   `export_kv_pages` + named-handle imports. The hermes-inferlet-side
   `block_cache::lookup_longest_prefix` is LIVE for lookup, but
   save-side is OFF on the chat path (disabled after an unbounded
   export-growth incident; see `src/block_cache.rs` top-of-module
   note). The net effect is: cross-session reuse for a chat request's
   tool-result body depends on whether a prior `/v1/pie/prompt-cache/
   chat-prefix:warm` pre-populated the shared prefix, not on Phase
   2.1's register route (which never had an import path anyway).

Together: a Phase-3.1 prefill-skip for tool-result bodies would need
to either (a) add SDK surface for mid-stream splice with K
re-rotation, or (b) turn the register route into something that
actually matters end-to-end (possibly via a chat-path save-on-export
re-enablement with scoping + LRU). Neither is load-bearing until we
know whether the existing layers already cover the workload. That's
what the Phase-3.0 APC pressure sweep measures.

**Rescoped Phase 3.0:** characterize the baseline. Sweep `N_users ×
L_tokens × pattern × {baseline|pinned}` synthetic scenarios and
measure per-round hit rate, TTFT, and the (baseline vs pinned) delta.
The `tool_result_tokens_imported` field in this divergence is how the
analyzer distinguishes "APC caught the whole prefix including the
body" from "APC stopped short at the body's leading block." Decision
points conditioned on sweep output:

- If baseline hit rate stays ≥ 90 % across realistic pressure →
  neither the register route nor a prefill-skip fast path adds
  measurable value for this workload. Phase 2.1 register gets flagged
  for retirement under task #28's next sync pass; Phase 3.1 scope
  collapses to documentation.
- If pinned arm shows a material delta vs baseline under pressure →
  Phase 2.1 has a real durability story; Phase 3.1 designs an import
  path that uses the pin. Mechanism TBD from (a)/(b) above.
- If both arms drop steeply under pressure AND the pinned arm is
  STRICTLY WORSE (pin steals capacity without offsetting benefit) →
  Phase 2.1 register is actively harmful; retire immediately.

**Back-compat:** zero by construction — the KV path is unchanged.
Telemetry field is `Option<u32>` with `skip_serializing_if`, so
responses to clients that don't know the field are byte-identical to
Phase 2.1.

**Upstream-worthy:** deferred. The detector alone has no obvious
standalone value for pieclaw; its value is tied to either a
prefill-skip mechanism (see above) or pressure-benchmark interpretation
(pie-hermes-specific workload). Task #28 revisits after the sweep
lands, informed by the retire-or-invest decision.
