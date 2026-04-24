use inferlet::forward::Forward;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

// ---------------------------------------------------------------------------
// Stored state types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefixCheckpoint {
    pub export_name: String,
    pub token_ids: Vec<u32>,
    pub kv_page_last_len: usize,
    pub content_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionKvState {
    pub session_id: String,
    pub export_name: String,
    /// format_chat output for this turn's INPUT messages (for prefix matching on next turn).
    /// NOT ctx.get_token_ids() — that includes generated tokens which don't round-trip.
    pub incoming_tokens: Vec<u32>,
    pub kv_page_last_len: usize,
    pub turn_count: u32,
    pub prefix_checkpoint_hash: String,
}

// ---------------------------------------------------------------------------
// Store keys
// ---------------------------------------------------------------------------

const PREFIX_CHECKPOINT_KEY: &str = "v2.prefix_checkpoint";
const SESSION_STATE_KEY: &str = "v2.session_state";
const ACTIVE_SESSION_KEY: &str = "v2.active_session_id";

// ---------------------------------------------------------------------------
// Hashing
// ---------------------------------------------------------------------------

pub fn sha256_hex(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    hex::encode(hasher.finalize())
}

/// Hash only the stable prefix of a reordered system prompt.
///
/// After `reorder_system_sections`, the dynamic `## Current Session Context`
/// section (carrying per-user user_name / chat_id / channel metadata emitted
/// by hermes-agent's `build_session_context_prompt`) sits at the end of the
/// system message. This function hashes everything *before* the first
/// dynamic header, so different users on the same bearer produce the same
/// stable-prefix hash — enabling prefix_checkpoint sharing across users
/// even when their per-session context differs.
///
/// Keep the marker list in sync with `DYNAMIC_SECTIONS` in
/// `prompt_render.rs`.
///
/// Callers must pass messages that have already been reordered.
pub fn hash_reordered_stable_prefix(messages: &[crate::types::ChatMessage]) -> String {
    let system_content = messages
        .iter()
        .find(|m| m.role == "system")
        .map(|m| m.content.as_str())
        .unwrap_or("");

    let dynamic_headers = ["\n## Current Session Context"];
    let cutoff = dynamic_headers
        .iter()
        .filter_map(|dh| system_content.find(dh))
        .min()
        .unwrap_or(system_content.len());

    let stable = &system_content[..cutoff];
    sha256_hex(stable)
}

// ---------------------------------------------------------------------------
// Token prefix matching
// ---------------------------------------------------------------------------

pub fn find_prefix_match_len(stored: &[u32], incoming: &[u32]) -> usize {
    stored
        .iter()
        .zip(incoming.iter())
        .take_while(|(a, b)| a == b)
        .count()
}

// ---------------------------------------------------------------------------
// Prefix checkpoint persistence
// ---------------------------------------------------------------------------

pub fn load_prefix_checkpoint() -> Option<PrefixCheckpoint> {
    let json = inferlet::store_get(PREFIX_CHECKPOINT_KEY)?;
    serde_json::from_str(&json).ok()
}

pub fn save_prefix_checkpoint(checkpoint: &PrefixCheckpoint) {
    if let Ok(json) = serde_json::to_string(checkpoint) {
        inferlet::store_set(PREFIX_CHECKPOINT_KEY, &json);
    }
}

// ---------------------------------------------------------------------------
// Session state persistence
// ---------------------------------------------------------------------------

pub fn load_session_state(session_id: &str) -> Option<SessionKvState> {
    let active = inferlet::store_get(ACTIVE_SESSION_KEY)?;
    if active != session_id {
        return None;
    }
    let json = inferlet::store_get(SESSION_STATE_KEY)?;
    let state: SessionKvState = serde_json::from_str(&json).ok()?;
    if state.session_id != session_id {
        return None;
    }
    Some(state)
}

pub fn save_session_state(state: &SessionKvState) {
    if let Ok(json) = serde_json::to_string(state) {
        inferlet::store_set(SESSION_STATE_KEY, &json);
        inferlet::store_set(ACTIVE_SESSION_KEY, &state.session_id);
    }
}

pub fn active_session_id() -> Option<String> {
    inferlet::store_get(ACTIVE_SESSION_KEY)
}

// ---------------------------------------------------------------------------
// Generation prompt suffix cache
// ---------------------------------------------------------------------------

const GEN_PROMPT_SUFFIX_KEY: &str = "v2.gen_prompt_suffix";

/// Cache the generation prompt suffix token IDs (computed once, reused per request).
/// For Qwen2.5: [151644, 77091, 198] = "<|im_start|>assistant\n"
pub fn load_gen_prompt_suffix() -> Option<Vec<u32>> {
    let json = inferlet::store_get(GEN_PROMPT_SUFFIX_KEY)?;
    serde_json::from_str(&json).ok()
}

pub fn save_gen_prompt_suffix(suffix: &[u32]) {
    if let Ok(json) = serde_json::to_string(suffix) {
        inferlet::store_set(GEN_PROMPT_SUFFIX_KEY, &json);
    }
}

// ---------------------------------------------------------------------------
// Export name helpers
// ---------------------------------------------------------------------------

pub fn session_export_name(session_id: &str) -> String {
    format!("session-kv:{}", session_id)
}

// ---------------------------------------------------------------------------
// Eviction
// ---------------------------------------------------------------------------

/// Evict the current session's KV pages (but NOT the prefix checkpoint).
///
/// Releases exactly one export (`state.export_name`). This is sufficient
/// because `handler::save_session_kv_state` enforces the invariant
/// "at most one live export per session at any time" by releasing the
/// prior turn's export before creating the new one. Historical versions
/// of this file (before the 2026-04-24 Phase-3.0 fix) leaked prior-turn
/// exports; any caller relying on `evict_session` to clean up ALL of a
/// session's exports should audit that assumption against the current
/// invariant — see `handler.rs` save_session_kv_state for the contract.
pub fn evict_session(model: &inferlet::Model) {
    if let Some(state) = load_current_session_state() {
        let queue = model.create_queue();
        queue.release_exported_kv_pages(&state.export_name);
        inferlet::store_delete(SESSION_STATE_KEY);
        inferlet::store_delete(ACTIVE_SESSION_KEY);
    }
}

fn load_current_session_state() -> Option<SessionKvState> {
    let json = inferlet::store_get(SESSION_STATE_KEY)?;
    serde_json::from_str(&json).ok()
}

// ---------------------------------------------------------------------------
// Import helpers
// ---------------------------------------------------------------------------

/// Import session KV pages, truncated to the incoming_tokens prefix.
/// The exported KV may include generated tokens beyond incoming_tokens;
/// we only import pages covering the incoming_tokens portion.
pub fn import_session_kv(
    model: &inferlet::Model,
    state: &SessionKvState,
) -> Option<inferlet::Context> {
    let prefix_len = state.incoming_tokens.len();
    if prefix_len == 0 {
        return None;
    }

    let queue = model.create_queue();

    // First try: import all exported pages
    let all_pages = queue.import_kv_pages(&state.export_name);
    if all_pages.is_empty() {
        return None;
    }

    let page_size = model.get_kv_page_size() as usize;
    if page_size == 0 {
        return None;
    }
    let pages_needed = (prefix_len + page_size - 1) / page_size;
    let truncated_last_len = prefix_len - (pages_needed - 1) * page_size;

    // If we have fewer pages than needed, the export was partial — give up
    if all_pages.len() < pages_needed {
        return None;
    }

    // Take only the pages covering the incoming_tokens prefix,
    // drop the rest (generated tokens' KV pages)
    let pages: Vec<_> = all_pages.into_iter().take(pages_needed).collect();

    Some(inferlet::Context::from_imported_state(
        model,
        pages,
        state.incoming_tokens.clone(),
        truncated_last_len,
    ))
}

pub fn import_prefix_checkpoint(
    model: &inferlet::Model,
    checkpoint: &PrefixCheckpoint,
) -> Option<inferlet::Context> {
    let queue = model.create_queue();
    let all_pages = queue.import_kv_pages(&checkpoint.export_name);
    if all_pages.is_empty() {
        return None;
    }

    // The lazy-save path exports the FULL request context (stable prefix +
    // dynamic + user + decoded tokens) because partial-slice export leaves
    // the exported pages unreachable after the source context drops.  Here
    // we truncate the imported pages back down to the stable-prefix length
    // recorded in the checkpoint metadata — same pattern as
    // `import_session_kv`.
    let prefix_len = checkpoint.token_ids.len();
    if prefix_len == 0 {
        return None;
    }

    let page_size = model.get_kv_page_size() as usize;
    if page_size == 0 {
        return None;
    }
    let pages_needed = (prefix_len + page_size - 1) / page_size;
    let truncated_last_len = if prefix_len % page_size == 0 {
        page_size
    } else {
        prefix_len - (pages_needed - 1) * page_size
    };

    if all_pages.len() < pages_needed {
        // Exported blob is smaller than the recorded prefix length — stale
        // metadata or a racing release; bail.
        return None;
    }

    let pages: Vec<_> = all_pages.into_iter().take(pages_needed).collect();

    Some(inferlet::Context::from_imported_state(
        model,
        pages,
        checkpoint.token_ids.clone(),
        truncated_last_len,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ChatMessage;

    fn sys(content: &str) -> ChatMessage {
        ChatMessage {
            role: "system".to_string(),
            content: content.to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    #[test]
    fn stable_prefix_hash_identical_when_only_session_context_differs() {
        // Same static persona, different per-user Current Session Context
        // blocks — stable-prefix hash must match so prefix_checkpoint is
        // reusable across users on one bearer.
        let alice = sys("static persona\n## Current Session Context\nuser=alice chat=general");
        let bob = sys("static persona\n## Current Session Context\nuser=bob chat=dm");
        assert_eq!(
            hash_reordered_stable_prefix(&[alice]),
            hash_reordered_stable_prefix(&[bob])
        );
    }

    #[test]
    fn stable_prefix_hash_differs_when_static_persona_differs() {
        let a = sys("persona A\n## Current Session Context\nuser=alice");
        let b = sys("persona B\n## Current Session Context\nuser=alice");
        assert_ne!(
            hash_reordered_stable_prefix(&[a]),
            hash_reordered_stable_prefix(&[b])
        );
    }

    #[test]
    fn stable_prefix_hash_falls_back_to_full_content_when_no_marker() {
        // If the caller never emitted `## Current Session Context` (e.g.
        // CLI-mode hermes-agent traffic), the whole content is hashed.
        let c = sys("static persona only, no session context section");
        let expected = sha256_hex("static persona only, no session context section");
        assert_eq!(hash_reordered_stable_prefix(&[c]), expected);
    }
}
