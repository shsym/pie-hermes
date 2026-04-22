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
