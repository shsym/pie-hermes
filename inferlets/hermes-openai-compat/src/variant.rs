//! POST /v1/pie/variant/export — pre-export a named variant KV segment.
//!
//! Request body: {"variant_name": "codex|google|none", "text": "..."}
//! Effect: prefills a standalone Context with `text`, exports its KV pages
//! under handle "hermes-variant-<variant_name>" on the underlying Queue.
//! Response: {"handle": "hermes-variant-<variant_name>", "token_count": N}.
//!
//! Local divergence from upstream pieclaw — see VENDOR_SOURCE.md.

use inferlet::{Context, Resource};
use serde::{Deserialize, Serialize};
use wstd::http::server::{Finished, Responder};
use wstd::http::{IntoBody, Response};

pub fn handle_name_for(variant: &str) -> String {
    format!("hermes-variant-{}", variant)
}

pub fn is_valid_variant(variant: &str) -> bool {
    matches!(variant, "codex" | "google" | "none")
}

#[derive(Deserialize)]
pub struct ExportRequest {
    pub variant_name: String,
    pub text: String,
}

#[derive(Serialize)]
pub struct ExportResponse {
    pub handle: String,
    pub token_count: usize,
}

pub async fn handle_export(body_bytes: Vec<u8>, responder: Responder) -> Finished {
    let req: ExportRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => {
            return crate::error_response(
                responder,
                400,
                &format!("Invalid JSON: {}", e),
            )
            .await;
        }
    };
    if !is_valid_variant(&req.variant_name) {
        return crate::error_response(
            responder,
            400,
            &format!(
                "variant_name must be one of codex|google|none, got {:?}",
                req.variant_name
            ),
        )
        .await;
    }

    let model = inferlet::get_auto_model();
    let mut ctx = Context::new(&model);
    if !req.text.is_empty() {
        ctx.fill_system(&req.text);
        ctx.flush().await;
    }
    let handle = handle_name_for(&req.variant_name);
    let token_count = ctx.get_token_ids().len();
    let kv_ptrs = ctx.get_kv_page_ptrs();
    // NOTE: export_resource_sync lives on Queue, not Model — the Context
    // carries the queue it was built on, so reuse that one.
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

    let resp = ExportResponse {
        handle: handle.clone(),
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
    fn handle_name_format() {
        assert_eq!(handle_name_for("codex"), "hermes-variant-codex");
        assert_eq!(handle_name_for("google"), "hermes-variant-google");
        assert_eq!(handle_name_for("none"), "hermes-variant-none");
    }

    #[test]
    fn rejects_unknown_variant() {
        assert!(!is_valid_variant("o1"));
        assert!(!is_valid_variant(""));
        assert!(!is_valid_variant("Codex")); // case-sensitive
    }

    #[test]
    fn accepts_known_variants() {
        assert!(is_valid_variant("codex"));
        assert!(is_valid_variant("google"));
        assert!(is_valid_variant("none"));
    }
}
