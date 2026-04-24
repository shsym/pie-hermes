//! Unit tests for the block_cache LRU index (task #47, item 12.1).
//!
//! Exercises the pure LRU functions (`lru_upsert_evict`, `lru_touch`)
//! directly — no `inferlet::store_*` or queue access needed. Compiled
//! via the crate's `rlib` target so `cargo test` on the host works.
//!
//! The store-coupled wrappers (`record_save_and_collect_evictions`,
//! `touch`, `configured_budget`) and full end-to-end eviction with
//! queue release are covered by the §6.3 accuracy probe on the pod.

use hermes_openai_compat::block_cache::{lru_touch, lru_upsert_evict, LruIndex};

#[test]
fn insertion_order_under_budget_no_eviction() {
    let mut idx = LruIndex::default();
    let budget = 1000;
    assert!(lru_upsert_evict(&mut idx, "a", 100, budget).is_empty());
    assert!(lru_upsert_evict(&mut idx, "b", 100, budget).is_empty());
    assert!(lru_upsert_evict(&mut idx, "c", 100, budget).is_empty());
    assert_eq!(idx.entries.len(), 3);
    let total: usize = idx.entries.iter().map(|e| e.pages).sum();
    assert_eq!(total, 300);
}

#[test]
fn over_budget_evicts_oldest_first() {
    let mut idx = LruIndex::default();
    let budget = 250;
    assert!(lru_upsert_evict(&mut idx, "a", 100, budget).is_empty());
    assert!(lru_upsert_evict(&mut idx, "b", 100, budget).is_empty());
    let evicted = lru_upsert_evict(&mut idx, "c", 100, budget);
    assert_eq!(evicted, vec!["a".to_string()]);
    let names: Vec<_> = idx.entries.iter().map(|e| e.name.clone()).collect();
    assert_eq!(names, vec!["b".to_string(), "c".to_string()]);
}

#[test]
fn touch_protects_from_eviction() {
    let mut idx = LruIndex::default();
    let budget = 250;
    lru_upsert_evict(&mut idx, "a", 100, budget);
    lru_upsert_evict(&mut idx, "b", 100, budget);
    // Without the touch, "a" would be oldest and evicted. Touch it so
    // "b" becomes oldest instead.
    lru_touch(&mut idx, "a");
    let evicted = lru_upsert_evict(&mut idx, "c", 100, budget);
    assert_eq!(evicted, vec!["b".to_string()]);
}

#[test]
fn over_budget_evicts_enough_to_fit() {
    let mut idx = LruIndex::default();
    let budget = 150;
    lru_upsert_evict(&mut idx, "a", 100, budget);
    lru_upsert_evict(&mut idx, "b", 100, budget);
    lru_upsert_evict(&mut idx, "c", 100, budget);
    let names: Vec<_> = idx.entries.iter().map(|e| e.name.clone()).collect();
    assert_eq!(names, vec!["c".to_string()]);
}

#[test]
fn never_evicts_the_just_saved_entry_even_if_oversized() {
    let mut idx = LruIndex::default();
    let budget = 50;
    // Single entry of 1000 pages exceeds budget but we don't evict it —
    // there's no older victim, so keep rather than empty the cache.
    let evicted = lru_upsert_evict(&mut idx, "huge", 1000, budget);
    assert!(evicted.is_empty());
    assert_eq!(idx.entries.len(), 1);
}

#[test]
fn upsert_preserves_access_ordering() {
    let mut idx = LruIndex::default();
    let budget = 1000;
    lru_upsert_evict(&mut idx, "a", 100, budget);
    lru_upsert_evict(&mut idx, "b", 100, budget);
    lru_upsert_evict(&mut idx, "c", 100, budget);
    // Re-save "a" — it moves to the newest slot.
    lru_upsert_evict(&mut idx, "a", 100, budget);
    let names: Vec<_> = idx.entries.iter().map(|e| e.name.clone()).collect();
    assert_eq!(names, vec!["b".to_string(), "c".to_string(), "a".to_string()]);
}

#[test]
fn upsert_updates_pages_on_duplicate() {
    let mut idx = LruIndex::default();
    let budget = 1000;
    lru_upsert_evict(&mut idx, "a", 100, budget);
    lru_upsert_evict(&mut idx, "a", 250, budget);
    assert_eq!(idx.entries.len(), 1);
    assert_eq!(idx.entries[0].pages, 250);
}

#[test]
fn touch_nonexistent_is_noop() {
    let mut idx = LruIndex::default();
    lru_touch(&mut idx, "does-not-exist");
    assert_eq!(idx.entries.len(), 0);
    assert_eq!(idx.counter, 0);
}

#[test]
fn serde_roundtrip_preserves_state() {
    let mut idx = LruIndex::default();
    lru_upsert_evict(&mut idx, "a", 100, 1000);
    lru_upsert_evict(&mut idx, "b", 200, 1000);
    let json = serde_json::to_string(&idx).unwrap();
    let round: LruIndex = serde_json::from_str(&json).unwrap();
    assert_eq!(round.entries.len(), 2);
    assert_eq!(round.counter, idx.counter);
}
