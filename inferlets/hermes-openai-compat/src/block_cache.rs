//! Block-level prefix cache inside the inferlet.
//!
//! ## STATUS (task #27, branch `task27-exp-naive-flip`)
//!
//! `lookup_longest_prefix` is **re-enabled** on this branch.  The
//! ⚠️ DISABLED rationale below is kept verbatim as the original bug
//! hypothesis; the task #27 naive-flip experiment tests whether the
//! `IMPORTED-CTX` branch added to pie-vllm's `vllm_sequence_tracker.py`
//! (commit 9118d46, 2026-04-11, ~18 hours AFTER the 2026-04-10 disable
//! below) already handles the sibling-walk corruption.  If the D3
//! reproducer at `scripts/bench-block-cache-repro.py` stays clean at
//! c=1/4/8/16, this flip is the shipping change.  If it reproduces, we
//! fall back to task #27 phase 2 path (a) — tracker-side fix.
//!
//! ## ⚠️ DISABLED — cross-instance history reconstruction is broken
//!
//! `lookup_longest_prefix` previously returned `None` unconditionally.
//! Measured behavior on D3 + D1-warmup (fresh A40 pod, 2026-04-10):
//!
//! - req0/req1/req3 are coherent, but **req2 is semantically off**
//!   even after capping match depth to leave a full-block tail.
//! - Lowering the cap changed the wrong-output content (math answer →
//!   "ticket prices report") but did not restore correctness.
//!
//! Root cause (traced in `pie-vllm/overlay/pie/src/pie_worker/vllm_sequence_tracker.py`):
//! the Python tracker reconstructs `full_prompt_token_ids` for a new
//! request by walking `blocks_per_req[i]` against `_active_requests`
//! and copying a sibling's `_token_history` slice of length
//! `effective_cached`.  When the warmup instance has already been
//! cleaned up (cleanup_instance → finish_by_identity), the walk instead
//! latches onto a still-live sibling that shares the **stable system
//! prefix** only (e.g. req0).  The tracker then copies the sibling's
//! divergent tokens as if they were the prefix, creating a mismatch
//! between Python's reconstructed history and the KV data actually
//! sitting in the imported pages.  vLLM attends to the wrong context
//! → output drifts to whatever the corrupted context suggests.
//!
//! The other working paths (`session_kv`, stable-only prefix
//! checkpoint, lazy prewarm) avoid this either by running warmup in the
//! same instance as the request, or by keeping the tail large enough
//! that the divergent copied history falls below `effective_cached`.
//!
//! A proper fix requires either (1) teaching the tracker to refuse the
//! sibling walk when block IDs beyond the shared prefix don't match, or
//! (2) sending the full prefix token IDs from the inferlet side so
//! Python never has to reconstruct.  Until one of those lands, block
//! cache must stay off in the hot path.  The save side is kept (it is
//! harmless: saving chain-hash → export-name entries doesn't affect
//! generation), so a future fix can re-enable lookup without touching
//! the write path.
//!
//! ## Design
//!
//! Each 16-token block (or whatever the engine's `kv_page_size` is) gets
//! a **rolling SHA256 chain hash** that captures the entire token sequence
//! up to that block.  Two requests whose reordered token sequences agree
//! for the first N blocks produce the same chain hash for each of those
//! blocks; any request can look up its block-i chain hash and find a
//! previously-exported KV slice that starts at position 0 and ends at
//! position (i+1) * page_size — regardless of which request originally
//! computed those blocks.
//!
//! ## Storage layout
//!
//! Two kinds of entries in `inferlet::store_*`:
//!
//! - `bc:blk:<hex_chain_hash>` → JSON `{ "req": "<final_hash>", "pages": N }`
//!   A lookup table entry.  N is the number of leading KV pages in the
//!   referenced request export that represent this block's prefix.
//!
//! - `bc:req:<hex_final_chain_hash>` is the **name** under which the
//!   actual exported KV pages live in the runtime resource registry
//!   (via `queue.export_kv_pages`).  It is NOT stored in
//!   `inferlet::store_*`; the runtime holds it.  We keep the chain hash
//!   as the stable identifier so we can always reconstruct the name.
//!
//! ## Save flow
//!
//! 1. Compute block chain hashes for every prefill block in the context.
//! 2. Export the FULL `ctx.kv_pages` (including decode tail) under
//!    `bc:req:<final_prefill_chain_hash>`.  The full-Vec export is
//!    required — partial-slice exports were observed to leave pages
//!    unreachable after the source context drops.  The decode tail is
//!    wasted KV but cleaned up when the export is released.
//! 3. For each prefill block i, write the lookup entry
//!    `bc:blk:<chain_hash_i>` pointing at `(req_name, i + 1)` IF the
//!    entry does not already exist.  Existing entries always win to
//!    preserve their (possibly shorter) backing exports.
//!
//! ## Lookup flow
//!
//! 1. Compute block chain hashes for every block of the incoming tokens
//!    (up to the reordered prefill length).
//! 2. Walk i = 0..num_blocks.  For each i, check
//!    `bc:blk:<chain_hash_i>`.  The LAST hit (highest i) gives us
//!    `(export_name, pages)` — the longest matching prefix of this
//!    request that is already cached.
//! 3. The caller imports `export_name`, truncates the pages to the
//!    first `pages` entries, and builds a Context.
//!
//! ## Constraints
//!
//! - Storage is unbounded in this initial implementation.  Add LRU
//!   eviction before production.
//! - The same physical KV page may be referenced by multiple lookup
//!   entries from different requests.  Releasing any one of those
//!   entries must not release the underlying pages as long as another
//!   entry still points at the same export name.

use inferlet::forward::Forward;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LookupEntry {
    req: String,
    pages: usize,
}

/// 32-byte chain hash.  `prev` is the hash of the previous block (zero-filled
/// for the very first block).  The block's tokens are hashed as little-endian
/// u32 bytes so the representation is deterministic across platforms.
pub fn chain_hash(prev: &[u8; 32], block_tokens: &[u32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(prev);
    for t in block_tokens {
        hasher.update(t.to_le_bytes());
    }
    let out = hasher.finalize();
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&out);
    arr
}

fn blk_key(h: &[u8; 32]) -> String {
    format!("bc:blk:{}", hex::encode(h))
}

fn req_name(h: &[u8; 32]) -> String {
    format!("bc:req:{}", hex::encode(h))
}

/// Walk `tokens` in `page_size` blocks and return the longest matching
/// prefix — the `(export_name, page_count)` of the deepest cache hit —
/// or None if no block-0 hit.
///
/// We stop at the first miss.  Gap-and-resume is not worth the SDK
/// complexity (mixed import + compute mid-context is not supported).
///
/// The deepest match is capped at `num_blocks - 1` so the caller always
/// has at least `page_size` tokens of fresh tail to re-prefill through
/// the model.  Importing the entire request would leave `ctx.fill_tokens`
/// with nothing to compute, and downstream paths assume the first forward
/// pass does at least one token of real work.
///
/// EXPERIMENTAL (task #27 `task27-exp-naive-flip`): this lookup was
/// disabled on 2026-04-10 (e26af34) under the hypothesis that Python's
/// `vllm_sequence_tracker.py` would reconstruct a stale sibling's history
/// on cross-instance import.  The `IMPORTED-CTX` tracker branch
/// (pie-vllm 9118d46, 2026-04-11) post-dates that disable and seeds
/// `token_history` with zero placeholders for `is_new AND effective_cached > 0`,
/// which — by code reading — bypasses the sibling walk entirely.  This
/// re-enable is the experiment that confirms or refutes that hypothesis
/// against the `bench-block-cache-repro.py` D3 workload.  If correctness
/// reproduces, fall back to a tracker-side fix (task #27 phase 2 path a).
pub fn lookup_longest_prefix(tokens: &[u32], page_size: usize) -> Option<(String, usize)> {
    if page_size == 0 || tokens.len() < 2 * page_size {
        // Need at least 2 blocks: one to match, one for fresh tail.
        return None;
    }
    let num_blocks = tokens.len() / page_size;
    let max_cached_pages = num_blocks - 1;

    let mut chain = [0u8; 32];
    let mut best: Option<(String, usize)> = None;
    for i in 0..num_blocks {
        let block = &tokens[i * page_size..(i + 1) * page_size];
        chain = chain_hash(&chain, block);
        let key = blk_key(&chain);
        let Some(json) = inferlet::store_get(&key) else {
            break;
        };
        let Ok(entry) = serde_json::from_str::<LookupEntry>(&json) else {
            break;
        };
        let capped_pages = entry.pages.min(max_cached_pages);
        if capped_pages > 0 {
            best = Some((entry.req, capped_pages));
        }
        // Deeper blocks can't beat the cap — stop walking.
        if entry.pages >= max_cached_pages {
            break;
        }
    }
    best
}

/// Save the current context's prefill blocks into the block cache.
///
/// `tokens` is the full prefill token sequence (reordered incoming_tokens
/// for this request, NOT including generated tokens).  The context's
/// `kv_pages` must cover at least `tokens.len()` tokens at the front
/// (the caller should only invoke this AFTER generation, when the ctx
/// contains prefill + decode pages).
///
/// Returns true iff at least one block was (re)saved.
pub fn save_ctx_blocks(
    ctx: &inferlet::Context,
    tokens: &[u32],
    page_size: usize,
) -> bool {
    if page_size == 0 {
        return false;
    }
    let num_blocks = tokens.len() / page_size;
    if num_blocks == 0 {
        return false;
    }
    // The ctx must cover at least `num_blocks` prefill pages.  If it
    // somehow doesn't (short request), abort quietly.
    if ctx.kv_pages.len() < num_blocks {
        return false;
    }

    // First, compute the chain hash chain over all prefill blocks and
    // find the final prefill hash — this is our export name.
    let mut chain = [0u8; 32];
    let mut chains: Vec<[u8; 32]> = Vec::with_capacity(num_blocks);
    for i in 0..num_blocks {
        let block = &tokens[i * page_size..(i + 1) * page_size];
        chain = chain_hash(&chain, block);
        chains.push(chain);
    }
    let final_chain = chains.last().copied().unwrap();
    let name = req_name(&final_chain);

    // Check whether the FINAL block's lookup entry already exists.  If
    // yes, an identical request prefix has already been saved — skip the
    // expensive re-export and only fill in any missing intermediate
    // lookup entries from this request's chain.
    let final_key = blk_key(&final_chain);
    let already_exported = inferlet::store_get(&final_key).is_some();

    if !already_exported {
        // Export the FULL ctx.kv_pages under our per-request name.  We
        // deliberately export the full Vec (not a slice) because the
        // runtime appears to require the full allocated set — see the
        // partial-slice-export crash notes in handler::save_prefix_checkpoint_from_ctx_if_needed.
        ctx.queue.export_kv_pages(&ctx.kv_pages, &name);
    }

    // Write one lookup entry per prefill block pointing at this
    // request's export.  Skip blocks that are ALREADY present — we
    // must not overwrite an existing entry that may reference a
    // different (still-alive) export.
    let mut wrote_any = !already_exported;
    for (i, ch) in chains.iter().enumerate() {
        let key = blk_key(ch);
        if inferlet::store_get(&key).is_some() {
            continue;
        }
        let entry = LookupEntry {
            req: name.clone(),
            pages: i + 1,
        };
        if let Ok(json) = serde_json::to_string(&entry) {
            inferlet::store_set(&key, &json);
            wrote_any = true;
        }
    }

    wrote_any
}

/// Import and truncate the matched prefix pages, building a Context
/// whose `kv_pages` cover exactly the first `pages` blocks of the
/// matched export.  Tokens are the prefix of the request up to
/// `pages * page_size`.
pub fn import_prefix(
    model: &inferlet::Model,
    export_name: &str,
    pages: usize,
    tokens: &[u32],
    page_size: usize,
) -> Option<inferlet::Context> {
    if pages == 0 {
        return None;
    }
    let need_tokens = pages * page_size;
    if tokens.len() < need_tokens {
        return None;
    }
    let queue = model.create_queue();
    let all_pages = queue.import_kv_pages(export_name);
    if all_pages.len() < pages {
        // Stale metadata pointing at a smaller (or released) export.
        return None;
    }
    let trimmed: Vec<_> = all_pages.into_iter().take(pages).collect();
    Some(inferlet::Context::from_imported_state(
        model,
        trimmed,
        tokens[..need_tokens].to_vec(),
        page_size,
    ))
}
