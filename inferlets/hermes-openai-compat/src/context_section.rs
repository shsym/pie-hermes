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

use inferlet::{Context, Resource};
use serde::{Deserialize, Serialize};
use wstd::http::server::{Finished, Responder};
use wstd::http::{IntoBody, Response};

pub fn handle_name_for(section_id: &str, body_hash: &str) -> String {
    // section_id may contain '#' or '/'; that is fine — it is opaque to the
    // engine's resource registry, and the body_hash suffix disambiguates.
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

pub fn save_metadata(meta: &SectionMetadata) {
    if let Ok(json) = serde_json::to_string(meta) {
        inferlet::store_set(&metadata_store_key(&meta.section_id, &meta.body_hash), &json);
    }
    // Side-table: latest hash for this section_id. Overwrites on re-register.
    inferlet::store_set(&latest_hash_store_key(&meta.section_id), &meta.body_hash);
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
    if req.body_hash.is_empty() {
        return crate::error_response(responder, 400, "body_hash must be non-empty").await;
    }

    let handle = handle_name_for(&req.section_id, &req.body_hash);

    // Empty body: skip engine round-trip, save empty metadata so the import
    // path can detect this section is registered (token_ids=[]). Pattern
    // mirrors the variant 'none' path in src/variant.rs:165.
    if req.body_text.is_empty() {
        save_metadata(&SectionMetadata {
            section_id: req.section_id,
            body_hash: req.body_hash,
            token_ids: vec![],
            kv_page_last_len: 0,
        });
        let resp = RegisterResponse {
            handle,
            token_count: 0,
        };
        let json = serde_json::to_string(&resp).unwrap_or_default();
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

    save_metadata(&SectionMetadata {
        section_id: req.section_id,
        body_hash: req.body_hash,
        token_ids,
        kv_page_last_len,
    });

    let resp = RegisterResponse {
        handle,
        token_count,
    };
    let json = serde_json::to_string(&resp).unwrap_or_default();
    let http = Response::builder()
        .header("Content-Type", "application/json")
        .body(json.into_body())
        .unwrap();
    responder.respond(http).await
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
}
