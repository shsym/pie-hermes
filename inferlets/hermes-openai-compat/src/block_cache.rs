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
//! Two-part state in `inferlet::store_*`:
//!
//! - `bc:lru:index` → JSON `{entries: [{name, pages}, ...]}` — the set of
//!   live exports and their pinned page counts.  Mutated on save only
//!   (warmup handler, low-concurrency path).
//! - `bc:lru:access:<name>` → decimal u64 — the per-export access token
//!   (read: "more-recent means larger").  Written on every save and on
//!   every chat-path `touch()`.  Each access write is a single atomic
//!   `store_set` — no read-modify-write on the hot path, so concurrent
//!   chat-path touches of different prefixes cannot race against each
//!   other.  See "Concurrency" below.
//! - `bc:lru:counter` → decimal u64 — monotonic token supplier.  Read
//!   once per touch/save under a read-incr-set pattern.  Races here
//!   collapse to a missed increment (two concurrent touches land on the
//!   same token value → tied access ordering), which is benign for LRU:
//!   the eviction path treats ties deterministically and the practical
//!   outcome is at worst a one-step ordering swap between entries that
//!   were already touched within the same microsecond.
//!
//! When total pinned pages exceed the configured budget (default:
//! unbounded), the entry with the smallest access token is popped and
//! its backing export released via `queue.release_exported_kv_pages`.
//! Lookup entries pointing at a released export are purged from
//! `bc:blk:*` during the same pass.
//!
//! The budget is a **launch-time** parameter, not a runtime knob:
//! `pie http --path ./hermes.wasm --port <n> -- --block-cache-budget-pages=<N>`.
//! Absence of the flag = unbounded (original shipping behavior).
//! Sweeping budgets — e.g. the memory-pressure benchmark — requires
//! relaunching the inferlet instance between arms.  This is deliberate:
//! runtime mutation of a pod-wide eviction budget is both a blast-radius
//! hazard (one caller can force-evict every other tenant's cached
//! prefixes) and a measurement hazard (LRU state and access tokens from
//! the prior arm contaminate the next one's steady-state).
//!
//! ## Concurrency
//!
//! The pie runtime kvs is a process-wide `DashMap` with serialized
//! per-command dispatch (see `pie/runtime/src/kvs.rs`), so each individual
//! `store_set` is atomic; but the SDK exposes no CAS primitive.  The
//! per-export access-key split above is specifically so the hot path
//! (`touch()`) is one `store_set` with no read-dependency: concurrent
//! touches of distinct names are independent; concurrent touches of the
//! same name last-write-wins to the same monotonic-ish token.
//!
//! The save-side (`record_save_and_collect_evictions`) still does a
//! load/modify/write on `bc:lru:index` and is therefore racy, but it
//! only runs from the warmup handler, which is effectively serial in
//! the memory-pressure benchmark.  An SDK-level CAS is tracked as a
//! follow-up.
//!
//! ## Error observability
//!
//! `save_session_kv_state` cannot log to stderr (wstd wedge, see
//! `tests/test_no_sync_eprintln.rs`).  Failures bump
//! `bc:stats:export_errors_total` and record the most recent message in
//! `bc:stats:last_export_error`; both are exposed via
//! `GET /v1/pie/block-cache/status` so an operator debugging a silent
//! regression has a signal to chase.
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
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LookupEntry {
    req: String,
    pages: usize,
}

/// Set of live exports tracked for LRU-eviction accounting.  Access
/// tokens live in a sibling map (`access_map`) rather than on the entry
/// so the hot-path `touch()` can bump a single scalar key without
/// rewriting the whole index — see the "Concurrency" section of the
/// module docstring.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LruIndex {
    pub entries: Vec<LruEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LruEntry {
    pub name: String,
    pub pages: usize,
}

/// Pure LRU upsert + eviction.  Does NOT touch the inferlet store or
/// any queue — returns the list of export names the caller must release.
/// Split out from the store-coupled wrapper so unit tests can exercise
/// ordering without the SDK.
///
/// `access_map` maps export name → access token.  An entry with no
/// access token is treated as older than any entry with one (access=0),
/// so freshly-inserted names with no touch history are the first victims.
/// `now` is the access token assigned to `name` on this call.
pub fn lru_upsert_evict(
    idx: &mut LruIndex,
    access_map: &mut HashMap<String, u64>,
    name: &str,
    pages: usize,
    now: u64,
    budget: usize,
) -> Vec<String> {
    access_map.insert(name.to_string(), now);

    let existing_pos = idx.entries.iter().position(|e| e.name == name);
    match existing_pos {
        Some(i) => idx.entries[i].pages = pages,
        None => idx.entries.push(LruEntry {
            name: name.to_string(),
            pages,
        }),
    }

    let mut evicted: Vec<String> = Vec::new();
    while idx.entries.len() > 1 {
        let total: usize = idx.entries.iter().map(|e| e.pages).sum();
        if total <= budget {
            break;
        }
        // Smallest access token wins the eviction lottery.  Tie-break by
        // name for determinism (concurrent touches can produce ties; we
        // do NOT want eviction order to depend on HashMap iteration).
        let oldest_idx = idx
            .entries
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let ta = access_map.get(&a.name).copied().unwrap_or(0);
                let tb = access_map.get(&b.name).copied().unwrap_or(0);
                ta.cmp(&tb).then_with(|| a.name.cmp(&b.name))
            })
            .map(|(i, _)| i)
            .expect("len > 1 checked above");
        let oldest = idx.entries.remove(oldest_idx);
        if oldest.name == name {
            // Never evict the just-saved entry.  Under tight budgets
            // where `pages > budget` for a single entry, we accept
            // temporary over-budget rather than empty the cache.
            idx.entries.insert(oldest_idx, oldest);
            break;
        }
        access_map.remove(&oldest.name);
        evicted.push(oldest.name);
    }
    evicted
}

/// Pure touch.  No-op if `name` is not in `idx` (stale lookup pointing
/// at a released export).
pub fn lru_touch(
    idx: &LruIndex,
    access_map: &mut HashMap<String, u64>,
    name: &str,
    now: u64,
) {
    if idx.entries.iter().any(|e| e.name == name) {
        access_map.insert(name.to_string(), now);
    }
}

const LRU_INDEX_KEY: &str = "bc:lru:index";
const LRU_COUNTER_KEY: &str = "bc:lru:counter";
const LRU_ACCESS_PREFIX: &str = "bc:lru:access:";
const EXPORT_ERR_TOTAL_KEY: &str = "bc:stats:export_errors_total";
const EXPORT_ERR_LAST_KEY: &str = "bc:stats:last_export_error";

/// Launch argument name recognized by `configured_budget`.  Parsed from
/// `inferlet::get_arguments()` in `--key=value` form.
const BUDGET_FLAG: &str = "--block-cache-budget-pages=";

fn access_key(name: &str) -> String {
    format!("{}{}", LRU_ACCESS_PREFIX, name)
}

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

/// Read every `bc:lru:access:*` entry into a `name → token` map.
/// O(all kvs keys) per call — same cost as the existing
/// `purge_lookup_entries_pointing_at`, and only run from the save path
/// (warmup-only).
fn load_access_map() -> HashMap<String, u64> {
    let mut out = HashMap::new();
    for key in inferlet::store_list_keys() {
        if let Some(name) = key.strip_prefix(LRU_ACCESS_PREFIX) {
            if let Some(v) = inferlet::store_get(&key) {
                if let Ok(token) = v.trim().parse::<u64>() {
                    out.insert(name.to_string(), token);
                }
            }
        }
    }
    out
}

/// Allocate the next access token.  Read-incr-set pattern on
/// `bc:lru:counter`; races collapse to a shared token value (benign —
/// see "Concurrency" in the module docstring).  `saturating_add` pins
/// at `u64::MAX` rather than wrapping, which under ~10^10 touches/year
/// is a practically unreachable ceiling (585-year headroom at
/// 10^9 touches/s).  If it ever saturates, all subsequent touches tie
/// at MAX, which degrades LRU to FIFO-on-save-order for new entries —
/// still correct, just loses freshness discrimination among the
/// saturated cohort.
fn allocate_access_token() -> u64 {
    let current = inferlet::store_get(LRU_COUNTER_KEY)
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0);
    let next = current.saturating_add(1);
    inferlet::store_set(LRU_COUNTER_KEY, &next.to_string());
    next
}

/// Read the configured budget (max pinned KV pages across all exports).
/// Parsed lazily from `inferlet::get_arguments()` on each call.  Returns
/// `usize::MAX` when the `--block-cache-budget-pages=<n>` flag is absent
/// — no eviction, original unbounded behavior.
///
/// Launch-time only: there is no runtime setter.  See the module
/// docstring for the rationale (blast-radius + measurement hygiene).
pub fn configured_budget() -> usize {
    parse_budget_from_args(&inferlet::get_arguments()).unwrap_or(usize::MAX)
}

/// Pure parser — exposed for unit tests so we can cover the flag-parsing
/// edge cases without a live runtime.  Accepts the flag in
/// `--block-cache-budget-pages=<n>` form anywhere in the argument vector;
/// returns `None` if the flag is absent, its value is empty, or parsing
/// fails (malformed values behave as "unset" rather than crashing the
/// inferlet — the launcher should have rejected them, but a silent
/// fall-through to unbounded is strictly safer than a panic in the
/// per-request path).
pub fn parse_budget_from_args(args: &[String]) -> Option<usize> {
    for arg in args {
        if let Some(rest) = arg.strip_prefix(BUDGET_FLAG) {
            return rest.trim().parse::<usize>().ok();
        }
    }
    None
}

/// Bump the access token of an existing LRU entry.  Called from the
/// chat-path handler after a successful `import_prefix` so hot prefixes
/// are preserved under eviction pressure.  One `store_set` on the hot
/// path — no read-modify-write on the shared index.  No-op if `name`
/// has been evicted and its access key removed (the subsequent touch
/// write is still harmless: the save path treats orphan access keys as
/// "entries with no index row" and ignores them).
pub fn touch(name: &str) {
    let token = allocate_access_token();
    inferlet::store_set(&access_key(name), &token.to_string());
}

/// Store-coupled wrapper around `lru_upsert_evict`.  Loads the index
/// and access map, applies the upsert with the configured budget,
/// writes the updated index, writes the new access token for `name`,
/// deletes access keys for evictees, and returns the list of export
/// names the caller must release.
fn record_save_and_collect_evictions(name: &str, pages: usize) -> Vec<String> {
    let mut idx = load_index();
    let mut accesses = load_access_map();
    let now = allocate_access_token();
    let evicted = lru_upsert_evict(&mut idx, &mut accesses, name, pages, now, configured_budget());
    save_index(&idx);
    inferlet::store_set(&access_key(name), &now.to_string());
    for evicted_name in &evicted {
        inferlet::store_delete(&access_key(evicted_name));
    }
    evicted
}

/// Record a (sync, stderr-forbidden) failure of the session-KV export
/// path.  Read-incr-set on the counter; last-message is overwritten with
/// a truncated copy so a noisy failure mode doesn't balloon the kvs.
/// Surfaced to operators via `/v1/pie/block-cache/status`.
pub fn record_export_error(msg: &str) {
    let current = inferlet::store_get(EXPORT_ERR_TOTAL_KEY)
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0);
    let next = current.saturating_add(1);
    inferlet::store_set(EXPORT_ERR_TOTAL_KEY, &next.to_string());
    let truncated: String = msg.chars().take(512).collect();
    inferlet::store_set(EXPORT_ERR_LAST_KEY, &truncated);
}

/// Summary of the current block-cache state — for
/// `/v1/pie/block-cache/status` introspection and test assertions.
/// Kept under the legacy `LruStatus` name so existing consumers don't
/// break; the type now carries additional fields that only appear in
/// the JSON output when populated.
#[derive(Debug, Clone, Serialize)]
pub struct LruStatus {
    pub budget_pages: Option<usize>,
    pub num_exports: usize,
    pub total_pinned_pages: usize,
    /// Monotonic access-token supplier value at the time of the status
    /// call.  Present for backward compatibility with operators that
    /// already diff this across scrapes.
    pub counter: u64,
    /// Cumulative count of `save_session_kv_state` export failures
    /// observed by `record_export_error`.  0 when no failures have
    /// ever been seen on this pod.
    pub export_errors_total: u64,
    /// The most recent export-error message (truncated to 512 chars),
    /// or `None` if no failures have been seen.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_export_error: Option<String>,
}

pub fn status() -> LruStatus {
    let idx = load_index();
    let total: usize = idx.entries.iter().map(|e| e.pages).sum();
    let budget = parse_budget_from_args(&inferlet::get_arguments());
    let counter = inferlet::store_get(LRU_COUNTER_KEY)
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0);
    let export_errors_total = inferlet::store_get(EXPORT_ERR_TOTAL_KEY)
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0);
    let last_export_error = inferlet::store_get(EXPORT_ERR_LAST_KEY);
    LruStatus {
        budget_pages: budget,
        num_exports: idx.entries.len(),
        total_pinned_pages: total,
        counter,
        export_errors_total,
        last_export_error,
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
    // Floor at 2×page_size rather than 1×: the deepest match is capped
    // to `num_blocks - 1` (see below) so we always re-prefill at least
    // one tail block.  That cap requires `num_blocks >= 2`, i.e.
    // `tokens.len() >= 2*page_size`.  A single-block prompt falls through
    // to the uncached path and pays the one-block prefill — acceptable
    // because the whole point of the cache is amortizing *shared*
    // prefixes across requests, and a one-block prompt carries no prefix
    // worth sharing.
    if page_size == 0 || tokens.len() < 2 * page_size {
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
