//! Block-level prefix cache inside the inferlet.
//!
//! ## Status
//!
//! Cross-session lookup is LIVE.  Chat requests that share a prefix with
//! a previously warmed sequence hit `lookup_longest_prefix` and import
//! the matching KV pages instead of re-prefilling.  Verified 2026-04-22
//! on A100-80GB with Qwen3-8B and Qwen2.5-72B-Instruct-AWQ via
//! `scripts/bench-block-cache-repro.py` (D3 @ c=1/4/8/16, 390 total
//! requests, 100% `fallback_reason=block_cache_hit`, zero
//! cross-contamination, zero engine errors).
//!
//! Save-side runs in the warmup handler (`/v1/pie/prompt-cache/chat-prefix:warm`)
//! only.  Calling `save_ctx_blocks` on every chat request floods the
//! runtime's export registry — each chat mutates the final chain hash
//! via its user turn, so `already_exported` is always false and
//! `queue.export_kv_pages` accumulates unbounded per-request entries,
//! producing `ExportSync FAILED / PointerNotAllocated` errors at c>=4.
//! Until an LRU / runtime-capacity strategy lands, chat-path save is
//! off; clients that want cross-session reuse should call the warmup
//! endpoint once per shared prefix.
//!
//! ## Historical — 2026-04-10 disable and why it no longer applies
//!
//! `lookup_longest_prefix` was disabled in commit `e26af34` (2026-04-10)
//! on the hypothesis that Python's `vllm_sequence_tracker.py` would
//! corrupt cross-instance imports by sibling-walking a still-live request
//! whose block IDs aliased the imported pages: the tracker would copy
//! the sibling's divergent `_token_history` slice as if it were the
//! imported prefix, and vLLM would attend to the wrong context.
//! Observed signature on a D3 + D1-warmup run at the time: req2 of 4
//! concurrent requests emitting "ticket prices report" instead of the
//! intended math answer.
//!
//! Eighteen hours later, pie-vllm commit `9118d46` (2026-04-11) added
//! an `IMPORTED-CTX` branch to the overlay tracker: when `is_new AND
//! existing_len == 0 AND effective_cached > 0 AND identity_key not in
//! token_history`, the tracker seeds `token_history` with
//! `[0] * effective_cached`.  `num_computed_tokens` lands at
//! `effective_cached`, so vLLM only embeds
//! `prompt_token_ids[num_computed_tokens:]` — the zero placeholders are
//! never read by the model.  The sibling-walk branch
//! (`vllm_sequence_tracker.py` lines 425-454) is unreachable for
//! cross-session imports under this guard.  The task #27 pod
//! verification exercised exactly that failure scenario and found zero
//! KV cross-contamination across 390 concurrent requests.
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
//! ## LRU eviction
//!
//! A single JSON index at `bc:lru:index` tracks every live export with
//! `(name, pages, access_counter)`. `save_ctx_blocks` upserts after
//! export; `touch` bumps on lookup-hit. When total pinned pages exceed
//! `bc:config:budget_pages` (default: unbounded), the oldest-access
//! entry is popped and its backing export released via
//! `queue.release_exported_kv_pages`. Lookup entries pointing at a
//! released export are purged from `bc:blk:*` during the same pass.
//!
//! Budget is set per request via a `?budget=<pages>` query param on the
//! warmup endpoint — this is the knob the memory-pressure benchmark
//! sweeps. Absence of the key = no eviction (original unbounded behavior).
//!
//! ## Constraints
//!
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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LruIndex {
    pub entries: Vec<LruEntry>,
    pub counter: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LruEntry {
    pub name: String,
    pub pages: usize,
    pub access: u64,
}

/// Pure LRU upsert + eviction.  Does NOT touch the inferlet store or
/// any queue — returns the list of export names the caller must release.
/// Split out from the store-coupled wrapper so unit tests can exercise
/// ordering without the SDK.
pub fn lru_upsert_evict(
    idx: &mut LruIndex,
    name: &str,
    pages: usize,
    budget: usize,
) -> Vec<String> {
    idx.counter = idx.counter.saturating_add(1);
    let access = idx.counter;

    let existing_pos = idx.entries.iter().position(|e| e.name == name);
    match existing_pos {
        Some(i) => {
            idx.entries[i].access = access;
            idx.entries[i].pages = pages;
        }
        None => idx.entries.push(LruEntry {
            name: name.to_string(),
            pages,
            access,
        }),
    }
    idx.entries.sort_by_key(|e| e.access);

    let mut evicted: Vec<String> = Vec::new();
    while idx.entries.len() > 1 {
        let total: usize = idx.entries.iter().map(|e| e.pages).sum();
        if total <= budget {
            break;
        }
        let oldest = idx.entries.remove(0);
        if oldest.name == name {
            idx.entries.insert(0, oldest);
            break;
        }
        evicted.push(oldest.name);
    }
    evicted
}

/// Pure touch (bump access) of an existing entry.  No-op if missing.
pub fn lru_touch(idx: &mut LruIndex, name: &str) {
    if let Some(i) = idx.entries.iter().position(|e| e.name == name) {
        idx.counter = idx.counter.saturating_add(1);
        idx.entries[i].access = idx.counter;
        idx.entries.sort_by_key(|e| e.access);
    }
}

const LRU_INDEX_KEY: &str = "bc:lru:index";
const BUDGET_KEY: &str = "bc:config:budget_pages";

fn load_index() -> LruIndex {
    inferlet::store_get(LRU_INDEX_KEY)
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_index(idx: &LruIndex) {
    if let Ok(s) = serde_json::to_string(idx) {
        inferlet::store_set(LRU_INDEX_KEY, &s);
    }
}

/// Read the configured budget (max pinned KV pages across all exports).
/// Returns `usize::MAX` when unset = no eviction.
pub fn configured_budget() -> usize {
    inferlet::store_get(BUDGET_KEY)
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(usize::MAX)
}

/// Set the budget (pages). Called by the warmup handler when ?budget=<n>
/// is present. Pass `None` to clear (restore unbounded behavior).
pub fn set_budget(pages: Option<usize>) {
    match pages {
        Some(n) => inferlet::store_set(BUDGET_KEY, &n.to_string()),
        None => inferlet::store_delete(BUDGET_KEY),
    }
}

/// Bump access counter for an existing LRU entry.  Called from handler
/// after a successful `import_prefix` on the chat path so hot prefixes
/// are preserved under eviction pressure.  No-op if `name` is not in the
/// index (stale lookup entry pointing at a released export).
pub fn touch(name: &str) {
    let mut idx = load_index();
    let before_counter = idx.counter;
    lru_touch(&mut idx, name);
    if idx.counter != before_counter {
        save_index(&idx);
    }
}

/// Store-coupled wrapper around `lru_upsert_evict`.  Loads the index,
/// applies the upsert with the configured budget, writes it back, and
/// returns the list of export names the caller must release.
fn record_save_and_collect_evictions(name: &str, pages: usize) -> Vec<String> {
    let mut idx = load_index();
    let evicted = lru_upsert_evict(&mut idx, name, pages, configured_budget());
    save_index(&idx);
    evicted
}

/// Summary of the current LRU state — for /v1/pie/block-cache/status
/// introspection and test assertions.
#[derive(Debug, Clone, Serialize)]
pub struct LruStatus {
    pub budget_pages: Option<usize>,
    pub num_exports: usize,
    pub total_pinned_pages: usize,
    pub counter: u64,
}

pub fn status() -> LruStatus {
    let idx = load_index();
    let total: usize = idx.entries.iter().map(|e| e.pages).sum();
    let budget = inferlet::store_get(BUDGET_KEY)
        .and_then(|s| s.trim().parse::<usize>().ok());
    LruStatus {
        budget_pages: budget,
        num_exports: idx.entries.len(),
        total_pinned_pages: total,
        counter: idx.counter,
    }
}

/// Remove all `bc:blk:*` lookup entries that reference a released
/// export. O(total lookup entries) per eviction — amortized over many
/// saves. Safe to call after release because stale entries would only
/// produce import-time misses, but cleaning keeps lookup fast.
fn purge_lookup_entries_pointing_at(released_name: &str) {
    for key in inferlet::store_list_keys() {
        if !key.starts_with("bc:blk:") {
            continue;
        }
        if let Some(json) = inferlet::store_get(&key) {
            if let Ok(entry) = serde_json::from_str::<LookupEntry>(&json) {
                if entry.req == released_name {
                    inferlet::store_delete(&key);
                }
            }
        }
    }
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
/// Re-enabled under task #27 after the pie-vllm `IMPORTED-CTX` tracker
/// branch (commit `9118d46`, 2026-04-11) made the original sibling-walk
/// corruption path unreachable for cross-session imports.  See the
/// "Historical" section in the module docstring.
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

    // Register in LRU index (upsert + access bump) and release any
    // entries that push us over the configured budget.  `ctx.kv_pages.len()`
    // is the pinned-page count because we exported the full Vec above.
    let pinned_pages = ctx.kv_pages.len();
    let evicted = record_save_and_collect_evictions(&name, pinned_pages);
    for evicted_name in &evicted {
        ctx.queue.release_exported_kv_pages(evicted_name);
        purge_lookup_entries_pointing_at(evicted_name);
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
