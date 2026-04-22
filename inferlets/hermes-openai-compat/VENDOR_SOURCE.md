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
