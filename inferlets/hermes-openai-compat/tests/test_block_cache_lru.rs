//! Unit tests for the block_cache LRU index (task #47, item 12.1).
//!
//! Exercises the pure LRU functions (`lru_upsert_evict`, `lru_touch`)
//! directly — no `inferlet::store_*` or queue access needed. Compiled
//! via the crate's `rlib` target so `cargo test` on the host works.
//!
//! The store-coupled wrappers (`record_save_and_collect_evictions`,
//! `touch`, `configured_budget`, `record_export_error`) and full
//! end-to-end eviction with queue release are covered by the §6.3
//! accuracy probe on the pod.

use std::collections::HashMap;

use hermes_openai_compat::block_cache::{
    LruEntry, LruIndex, lru_touch, lru_upsert_evict, parse_budget_from_args,
};

/// Helper: construct a fresh `(LruIndex, access_map)` pair.
fn fresh() -> (LruIndex, HashMap<String, u64>) {
    (LruIndex::default(), HashMap::new())
}

#[test]
fn insertion_order_under_budget_no_eviction() {
    let (mut idx, mut acc) = fresh();
    let budget = 1000;
    assert!(lru_upsert_evict(&mut idx, &mut acc, "a", 100, 1, budget).is_empty());
    assert!(lru_upsert_evict(&mut idx, &mut acc, "b", 100, 2, budget).is_empty());
    assert!(lru_upsert_evict(&mut idx, &mut acc, "c", 100, 3, budget).is_empty());
    assert_eq!(idx.entries.len(), 3);
    let total: usize = idx.entries.iter().map(|e| e.pages).sum();
    assert_eq!(total, 300);
}

#[test]
fn over_budget_evicts_oldest_first() {
    let (mut idx, mut acc) = fresh();
    let budget = 250;
    assert!(lru_upsert_evict(&mut idx, &mut acc, "a", 100, 1, budget).is_empty());
    assert!(lru_upsert_evict(&mut idx, &mut acc, "b", 100, 2, budget).is_empty());
    let evicted = lru_upsert_evict(&mut idx, &mut acc, "c", 100, 3, budget);
    assert_eq!(evicted, vec!["a".to_string()]);
    let names: Vec<_> = idx.entries.iter().map(|e| e.name.clone()).collect();
    assert_eq!(names, vec!["b".to_string(), "c".to_string()]);
    // Evicted entry's access key is dropped too.
    assert!(!acc.contains_key("a"));
}

#[test]
fn touch_protects_from_eviction() {
    let (mut idx, mut acc) = fresh();
    let budget = 250;
    lru_upsert_evict(&mut idx, &mut acc, "a", 100, 1, budget);
    lru_upsert_evict(&mut idx, &mut acc, "b", 100, 2, budget);
    // Without the touch, "a" would be oldest and evicted. Touch it so
    // "b" becomes oldest instead.
    lru_touch(&idx, &mut acc, "a", 3);
    let evicted = lru_upsert_evict(&mut idx, &mut acc, "c", 100, 4, budget);
    assert_eq!(evicted, vec!["b".to_string()]);
}

#[test]
fn touch_on_unknown_entry_is_noop() {
    let (idx, mut acc) = fresh();
    lru_touch(&idx, &mut acc, "ghost", 42);
    assert!(acc.is_empty());
}

#[test]
fn over_budget_evicts_enough_to_fit() {
    let (mut idx, mut acc) = fresh();
    let budget = 150;
    lru_upsert_evict(&mut idx, &mut acc, "a", 100, 1, budget);
    lru_upsert_evict(&mut idx, &mut acc, "b", 100, 2, budget);
    lru_upsert_evict(&mut idx, &mut acc, "c", 100, 3, budget);
    let names: Vec<_> = idx.entries.iter().map(|e| e.name.clone()).collect();
    assert_eq!(names, vec!["c".to_string()]);
}

#[test]
fn never_evicts_the_just_saved_entry_even_if_oversized() {
    let (mut idx, mut acc) = fresh();
    let budget = 50;
    // Single entry of 1000 pages exceeds budget but we don't evict it —
    // there's no older victim, so keep rather than empty the cache.
    let evicted = lru_upsert_evict(&mut idx, &mut acc, "huge", 1000, 1, budget);
    assert!(evicted.is_empty());
    assert_eq!(idx.entries.len(), 1);
}

#[test]
fn upsert_updates_pages_on_duplicate() {
    let (mut idx, mut acc) = fresh();
    let budget = 1000;
    lru_upsert_evict(&mut idx, &mut acc, "a", 100, 1, budget);
    lru_upsert_evict(&mut idx, &mut acc, "a", 250, 2, budget);
    assert_eq!(idx.entries.len(), 1);
    assert_eq!(idx.entries[0].pages, 250);
    assert_eq!(acc.get("a"), Some(&2));
}

#[test]
fn upsert_without_access_treats_untouched_entries_as_oldest() {
    // An entry with no access-map row is treated as older (access=0)
    // than any entry with one.  This covers the first-save-after-restart
    // path and entries left over from prior runs with no sibling access
    // key.
    let (mut idx, mut acc) = fresh();
    idx.entries.push(LruEntry {
        name: "stale".into(),
        pages: 100,
    });
    let budget = 150;
    let evicted = lru_upsert_evict(&mut idx, &mut acc, "fresh", 100, 1, budget);
    assert_eq!(evicted, vec!["stale".to_string()]);
}

#[test]
fn eviction_is_deterministic_under_tied_access() {
    // Two entries at the same access token (the race outcome described
    // in the module's "Concurrency" note) — tiebreak is by name asc,
    // so "a" loses to "b" on the next eviction round.  Budget is 250
    // (not 150) so "a" and "b" coexist until "c" forces a single
    // eviction rather than cascading through both.
    let (mut idx, mut acc) = fresh();
    let budget = 250;
    lru_upsert_evict(&mut idx, &mut acc, "b", 100, 5, budget);
    lru_upsert_evict(&mut idx, &mut acc, "a", 100, 5, budget);
    let evicted = lru_upsert_evict(&mut idx, &mut acc, "c", 100, 6, budget);
    assert_eq!(evicted, vec!["a".to_string()]);
}

#[test]
fn parse_budget_from_args_handles_common_cases() {
    let s = |v: &str| v.to_string();

    // Absent → None (= unbounded at the caller).
    assert_eq!(parse_budget_from_args(&[]), None);
    assert_eq!(
        parse_budget_from_args(&[s("--other=1"), s("--another")]),
        None
    );

    // Present as `--block-cache-budget-pages=<n>` → parsed.
    assert_eq!(
        parse_budget_from_args(&[s("--block-cache-budget-pages=512")]),
        Some(512)
    );
    // Works anywhere in the vector.
    assert_eq!(
        parse_budget_from_args(&[
            s("--first"),
            s("--block-cache-budget-pages=2048"),
            s("--last"),
        ]),
        Some(2048)
    );
    // Whitespace around the integer is tolerated.
    assert_eq!(
        parse_budget_from_args(&[s("--block-cache-budget-pages=  64 ")]),
        Some(64)
    );

    // Malformed value → None, not a panic.  Silent fall-through is
    // strictly safer than a per-request crash if the launcher lets a
    // bad value through.
    assert_eq!(
        parse_budget_from_args(&[s("--block-cache-budget-pages=abc")]),
        None
    );
    assert_eq!(
        parse_budget_from_args(&[s("--block-cache-budget-pages=")]),
        None
    );

    // Space-separated form (`--flag <val>`) is NOT supported — the
    // parser expects `--flag=<val>`.  Documenting the non-support so a
    // future contributor doesn't add it inconsistently.
    assert_eq!(
        parse_budget_from_args(&[s("--block-cache-budget-pages"), s("512")]),
        None
    );
}

#[test]
fn serde_roundtrip_preserves_state() {
    let (mut idx, mut acc) = fresh();
    lru_upsert_evict(&mut idx, &mut acc, "a", 100, 1, 1000);
    lru_upsert_evict(&mut idx, &mut acc, "b", 200, 2, 1000);
    let json = serde_json::to_string(&idx).unwrap();
    let round: LruIndex = serde_json::from_str(&json).unwrap();
    assert_eq!(round.entries.len(), 2);
    assert_eq!(round.entries[0].name, "a");
    assert_eq!(round.entries[1].pages, 200);
}
