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
