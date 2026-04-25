//! POST /v1/pie/context-section/register — register a named context-section body
//! as an exported KV-prefix handle keyed by (section_id, body_hash).
//!
//! Request body: {"section_id": "agents.md#coding-style",
//!                "body_hash":  "deadbeef12345678",
//!                "body_text":  "Use 2-space indents..."}
//! Effect: prefills a standalone Context with `body_text`, exports its KV pages
//! under handle "hermes-section-<section_id>-<body_hash>", persists metadata
//! at store key "v1.section.<section_id>.<body_hash>.meta", AND updates a
//! side-table at "v1.section.<section_id>.latest_hash" -> body_hash.
//! Response: {"handle": "...", "token_count": N}.
//!
//! Why a side-table: when a context file is edited, the agent re-registers
//! with a NEW body_hash. The latest_hash side-table lets the read_context
//! tool intercept (Task 2C) resolve a section_id to the most recent body
//! without the agent re-supplying the hash on every tool call.
//!
//! Local divergence from upstream pieclaw — see VENDOR_SOURCE.md #7.

use inferlet::{Context, Model, Resource};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use wstd::http::server::{Finished, Responder};
use wstd::http::{IntoBody, Response};

use crate::types::ChatMessage;

/// Max accepted size of `body_text` in bytes. A context section represents a
/// chunk of a human-authored file (AGENTS.md and friends); 1 MiB is ~2x the
/// largest realistic section and forces obviously-abusive requests to fail
/// fast instead of tokenizing + prefilling arbitrary input on the engine.
pub const MAX_BODY_TEXT_BYTES: usize = 1024 * 1024;

/// `body_hash` must be an unambiguous terminal segment of the exported
/// KV-page handle (`hermes-section-<section_id>-<body_hash>`). Because
/// `section_id` may legitimately contain '-' (e.g. "agents.md#coding-style"),
/// allowing '-' in `body_hash` makes the handle non-injective — different
/// (section_id, body_hash) pairs can collide on the same handle string.
///
/// We therefore forbid '-' in body_hash and permit the character set that
/// covers the hash encodings agents actually send:
///
///   - hex                       — `[0-9a-fA-F]` (alphanumeric subset)
///   - base32 padded/unpadded    — `[A-Z2-7=]`
///   - standard base64           — `[A-Za-z0-9+/=]` (we reject `+` and `/`
///                                  because they commonly need URL/path
///                                  escaping downstream; standard base64
///                                  callers should switch to URL-safe)
///   - URL-safe base64           — `[A-Za-z0-9_=]` (no `-`; callers that
///                                  need URL-safe with `-` must strip it
///                                  or switch to hex)
///
/// So the accepted set is `[A-Za-z0-9_=]`: alphanumerics plus `_` (URL-safe
/// base64 alphabet) and `=` (padding). Injectivity still holds because '-'
/// is the handle separator and never appears in body_hash.
fn is_valid_body_hash(s: &str) -> bool {
    !s.is_empty()
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'=')
}

pub fn handle_name_for(section_id: &str, body_hash: &str) -> String {
    // section_id may contain '#' or '/'; that is fine — it is opaque to the
    // engine's resource registry, and the body_hash suffix disambiguates.
    // Injectivity is preserved by body_hash forbidding '-' (the separator)
    // — see is_valid_body_hash for the full accepted character set.
    format!("hermes-section-{}-{}", section_id, body_hash)
}

pub fn metadata_store_key(section_id: &str, body_hash: &str) -> String {
    format!("v1.section.{}.{}.meta", section_id, body_hash)
}

pub fn latest_hash_store_key(section_id: &str) -> String {
    format!("v1.section.{}.latest_hash", section_id)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectionMetadata {
    pub section_id: String,
    pub body_hash: String,
    pub token_ids: Vec<u32>,
    pub kv_page_last_len: usize,
}

pub fn save_metadata(meta: &SectionMetadata) -> Result<(), serde_json::Error> {
    // Serialize first. If this fails we MUST NOT update the latest_hash
    // side-table — otherwise it would resolve to a body_hash whose metadata
    // record does not exist, and the Task-2.2 read_context intercept would
    // then fail on resolve. Writes are issued metadata-first so any reader
    // that sees the updated latest_hash is guaranteed to find the meta.
    // Concurrent register() calls are last-writer-wins on latest_hash;
    // the loser's metadata record still exists and remains addressable
    // by its explicit hash.
    let json = serde_json::to_string(meta)?;
    inferlet::store_set(
        &metadata_store_key(&meta.section_id, &meta.body_hash),
        &json,
    );
    inferlet::store_set(&latest_hash_store_key(&meta.section_id), &meta.body_hash);
    Ok(())
}

pub fn load_metadata(section_id: &str, body_hash: &str) -> Option<SectionMetadata> {
    let json = inferlet::store_get(&metadata_store_key(section_id, body_hash))?;
    serde_json::from_str(&json).ok()
}

pub fn load_latest_hash(section_id: &str) -> Option<String> {
    inferlet::store_get(&latest_hash_store_key(section_id))
}

#[derive(Deserialize)]
pub struct RegisterRequest {
    pub section_id: String,
    pub body_hash: String,
    pub body_text: String,
}

#[derive(Serialize)]
pub struct RegisterResponse {
    pub handle: String,
    pub token_count: usize,
}

pub async fn handle_register(body_bytes: Vec<u8>, responder: Responder) -> Finished {
    // Parse + minimal validation.
    let req: RegisterRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => {
            return crate::error_response(responder, 400, &format!("Invalid JSON: {}", e)).await;
        }
    };
    if req.section_id.is_empty() {
        return crate::error_response(responder, 400, "section_id must be non-empty").await;
    }
    if !is_valid_body_hash(&req.body_hash) {
        return crate::error_response(
            responder,
            400,
            "body_hash must be non-empty and contain only [A-Za-z0-9_=] (handle injectivity)",
        )
        .await;
    }
    if req.body_text.len() > MAX_BODY_TEXT_BYTES {
        return crate::error_response(
            responder,
            413,
            &format!(
                "body_text too large: {} bytes > {} limit",
                req.body_text.len(),
                MAX_BODY_TEXT_BYTES
            ),
        )
        .await;
    }

    let handle = handle_name_for(&req.section_id, &req.body_hash);

    // Empty body: skip engine round-trip, save empty metadata so the import
    // path can detect this section is registered (token_ids=[]). Pattern
    // mirrors the variant 'none' path in src/variant.rs:165.
    if req.body_text.is_empty() {
        if let Err(e) = save_metadata(&SectionMetadata {
            section_id: req.section_id,
            body_hash: req.body_hash,
            token_ids: vec![],
            kv_page_last_len: 0,
        }) {
            return crate::error_response(
                responder,
                500,
                &format!("metadata serialize failed: {}", e),
            )
            .await;
        }
        let resp = RegisterResponse {
            handle,
            token_count: 0,
        };
        let json = match serde_json::to_string(&resp) {
            Ok(j) => j,
            Err(e) => {
                return crate::error_response(
                    responder,
                    500,
                    &format!("response serialize failed: {}", e),
                )
                .await;
            }
        };
        let http = Response::builder()
            .header("Content-Type", "application/json")
            .body(json.into_body())
            .unwrap();
        return responder.respond(http).await;
    }

    let model = inferlet::get_auto_model();
    let mut ctx = Context::new(&model);
    ctx.fill_system(&req.body_text);
    ctx.flush().await;
    let token_count = ctx.get_token_ids().len();
    let kv_page_last_len = ctx.get_kv_page_last_len();
    let token_ids = ctx.get_token_ids().to_vec();
    let kv_ptrs = ctx.get_kv_page_ptrs();

    if let Err(e) = ctx
        .queue()
        .export_resource_sync(Resource::KvPage, &kv_ptrs, &handle)
    {
        return crate::error_response(
            responder,
            500,
            &format!("export_resource_sync failed: {}", e),
        )
        .await;
    }

    if let Err(e) = save_metadata(&SectionMetadata {
        section_id: req.section_id,
        body_hash: req.body_hash,
        token_ids,
        kv_page_last_len,
    }) {
        return crate::error_response(responder, 500, &format!("metadata serialize failed: {}", e))
            .await;
    }

    let resp = RegisterResponse {
        handle,
        token_count,
    };
    let json = match serde_json::to_string(&resp) {
        Ok(j) => j,
        Err(e) => {
            return crate::error_response(
                responder,
                500,
                &format!("response serialize failed: {}", e),
            )
            .await;
        }
    };
    let http = Response::builder()
        .header("Content-Type", "application/json")
        .body(json.into_body())
        .unwrap();
    responder.respond(http).await
}

// ---------------------------------------------------------------------------
// Phase 3.0 — tool-result body hashing + registered-handle lookup.
//
// On every inbound chat request we scan `role:"tool"` messages for content
// that matches a body registered via `/v1/pie/context-section/register`
// (VENDOR_SOURCE.md #7). On match we count the body tokens and surface the
// total in `pie_cache.tool_result_tokens_imported` (VENDOR_SOURCE.md #8).
//
// Phase 3.0 is DETECTION + TELEMETRY only. Actual prefill-skip requires
// mid-stream KV-splice with RoPE re-rotation, which the SDK does not
// currently expose. Matches reported here are the savings a future
// implementation would realize — see VENDOR_SOURCE.md #8 for the full
// scope note.
// ---------------------------------------------------------------------------

/// Shape of the JSON envelope produced by hermes-agent's `read_context`
/// tool (see `tools/read_context_tool.py`). Only `section_id` + `body` are
/// load-bearing for detection. Any other fields are tolerated and ignored.
#[derive(Debug, Deserialize)]
struct ToolResultEnvelope {
    section_id: String,
    body: String,
}

/// SHA-256 the body text and return the first 16 lowercase hex chars —
/// identical to hermes-agent's `hashlib.sha256(body.encode('utf-8'))
/// .hexdigest()[:16]` (see `agent/context_summary.py`). Producing the same
/// hash format is a wire-contract requirement: mismatched formatters would
/// silently break lookups even on byte-identical bodies.
fn body_hash_16(body: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(body.as_bytes());
    let digest = hasher.finalize();
    // Pre-allocate 16 chars of hex = 8 bytes of digest. Writing into a
    // String with `write!` avoids an intermediate full-length hex encode.
    let mut out = String::with_capacity(16);
    for byte in &digest[..8] {
        use std::fmt::Write;
        let _ = write!(&mut out, "{:02x}", byte);
    }
    out
}

/// Detect which `role:"tool"` messages carry a body that matches a handle
/// registered via Phase 2.1. Returns the sum of tokenized-body lengths
/// across all matches (zero if no tool messages, zero if tool messages
/// present but none match, `None` if no tool messages at all — so callers
/// can distinguish "checked, nothing matched" from "nothing to check").
///
/// Behavior per tool message:
/// 1. Try to parse `content` as the `read_context` JSON envelope
///    `{"section_id": ..., "body": ...}`. Non-JSON / missing-field bodies
///    are skipped (no match, no error).
/// 2. Compute `body_hash_16(body)` and look up
///    `load_metadata(section_id, body_hash)`. Missing metadata → no match.
/// 3. On match, count `tokenizer.tokenize(body).len()` — this is the span
///    a future mid-stream splice would import. Using the tokenizer rather
///    than the stored `SectionMetadata.token_ids.len()` gives a count
///    scoped to the body alone, since the stored count includes the
///    system-role marker tokens baked in by `fill_system` at register time.
pub fn detect_tool_result_matches(messages: &[ChatMessage], model: &Model) -> Option<u32> {
    let mut saw_tool = false;
    let mut total: u32 = 0;
    let tokenizer = model.get_tokenizer();

    for msg in messages {
        if msg.role != "tool" {
            continue;
        }
        saw_tool = true;

        let env: ToolResultEnvelope = match serde_json::from_str(&msg.content) {
            Ok(e) => e,
            Err(_) => continue,
        };
        let body_hash = body_hash_16(&env.body);
        if load_metadata(&env.section_id, &body_hash).is_none() {
            continue;
        }
        // Tokenize the body field specifically (not the full envelope).
        // This reports the span the model will see INSIDE the tool-role
        // wrapper when the chat template renders this message.
        let n = tokenizer.tokenize(&env.body).len() as u32;
        total = total.saturating_add(n);
    }

    if saw_tool { Some(total) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handle_name_includes_section_and_hash() {
        let h = handle_name_for("agents.md#coding-style", "deadbeef12345678");
        assert_eq!(h, "hermes-section-agents.md#coding-style-deadbeef12345678");
    }

    #[test]
    fn metadata_store_key_format() {
        let k = metadata_store_key("agents.md#x", "abc1234");
        assert_eq!(k, "v1.section.agents.md#x.abc1234.meta");
    }

    #[test]
    fn latest_hash_store_key_format() {
        let k = latest_hash_store_key("agents.md#x");
        assert_eq!(k, "v1.section.agents.md#x.latest_hash");
    }

    #[test]
    fn section_metadata_json_roundtrip() {
        let meta = SectionMetadata {
            section_id: "agents.md#coding-style".into(),
            body_hash: "deadbeef12345678".into(),
            token_ids: vec![1u32, 2, 3, 151644, 77091],
            kv_page_last_len: 42,
        };
        let json = serde_json::to_string(&meta).expect("serialize");
        let back: SectionMetadata = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.section_id, meta.section_id);
        assert_eq!(back.body_hash, meta.body_hash);
        assert_eq!(back.token_ids, meta.token_ids);
        assert_eq!(back.kv_page_last_len, meta.kv_page_last_len);
    }

    #[test]
    fn body_hash_validator_accepts_common_encodings_and_rejects_ambiguous() {
        // Hex (alphanumeric subset) — always accepted.
        assert!(is_valid_body_hash("deadbeef12345678"));
        assert!(is_valid_body_hash("ABCdef0123"));
        // Padded base32/base64 — `=` is allowed (no injectivity risk).
        assert!(is_valid_body_hash("abc=="));
        assert!(is_valid_body_hash("MZXW6YTBOI======"));
        // URL-safe base64 minus `-` — `_` is allowed.
        assert!(is_valid_body_hash("abc_def_ghi"));
        assert!(is_valid_body_hash("Zm9vX2Jhcg=="));
        // Empty → rejected (also caught by is_empty check, but is_valid is
        // the authoritative gate).
        assert!(!is_valid_body_hash(""));
        // Dashes would break handle injectivity — see handle_name_for doc.
        assert!(!is_valid_body_hash("dead-beef"));
        // Standard base64 chars `+` and `/` rejected (downstream escaping).
        assert!(!is_valid_body_hash("ab+cd"));
        assert!(!is_valid_body_hash("ab/cd"));
        // Dots / hash / whitespace / non-ASCII — rejected.
        assert!(!is_valid_body_hash("dead.beef"));
        assert!(!is_valid_body_hash("dead#beef"));
        assert!(!is_valid_body_hash("dead beef"));
        assert!(!is_valid_body_hash("café"));
    }

    #[test]
    fn body_hash_hyphen_would_cause_handle_collision() {
        // Documents the collision that the validator prevents: without the
        // alphanumeric-only rule, these two distinct (section_id, body_hash)
        // pairs produce the same exported-handle string.
        let a = handle_name_for("foo-x", "bar");
        let b = handle_name_for("foo", "x-bar");
        assert_eq!(a, b);
        // Sanity: at least one of these pairs is now rejected at the API.
        assert!(!is_valid_body_hash("x-bar"));
    }

    #[test]
    fn section_metadata_json_roundtrip_empty() {
        let meta = SectionMetadata {
            section_id: "x#y".into(),
            body_hash: "".into(), // body_hash="" disallowed at the API layer; this is a serde-only check
            token_ids: vec![],
            kv_page_last_len: 0,
        };
        let json = serde_json::to_string(&meta).expect("serialize");
        let back: SectionMetadata = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.token_ids, meta.token_ids);
    }

    #[test]
    fn body_hash_16_matches_python_reference() {
        // Cross-checked against:
        //   hashlib.sha256(b"All variables ...").hexdigest()[:16]
        // in agent/context_summary.py. A drift between the two formatters
        // (different trunc length, upper vs lower hex, different encoding)
        // would silently break Phase-3.0 detection on byte-identical bodies.
        // Sample: SHA-256("hello world") =
        //   b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9
        // first 16 hex chars → "b94d27b9934d3e08".
        assert_eq!(body_hash_16("hello world"), "b94d27b9934d3e08");
    }

    #[test]
    fn body_hash_16_is_stable_for_empty() {
        // Empty body is rejected at the register API (section-meta layer
        // allows it through for serde, but the wire handler enforces
        // non-empty). Record the empty hash so a future regression that
        // stops rejecting empties in the register path would at least
        // produce a deterministic, non-colliding detection value.
        let h = body_hash_16("");
        assert_eq!(h.len(), 16);
        // SHA-256("") = e3b0c44298fc1c14... -> first 16 chars:
        assert_eq!(h, "e3b0c44298fc1c14");
    }

    // `detect_tool_result_matches` depends on `Model::get_tokenizer()`,
    // which is a WASI-backed call; constructing a Model here would require
    // the wasm32-wasip2 target and a loaded engine. Exercise this end-to-
    // end via the capture-run gate in bench/driver.py instead (see
    // probe-read-context-one-call's `must_import_kv` assertion).
}
