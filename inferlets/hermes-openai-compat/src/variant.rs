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

// ---------------------------------------------------------------------------
// Variant metadata — persisted alongside the exported KV pages so the import
// path can reconstruct a Context via `Context::from_imported_state`.
// Pattern mirrors `session_cache::PrefixCheckpoint`.
// ---------------------------------------------------------------------------

/// Metadata stored alongside a variant's exported KV pages.  Required to
/// reconstruct a Context via `from_imported_state` on import.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantMetadata {
    /// Tokens the context was prefilled with before export.
    pub token_ids: Vec<u32>,
    /// Final KV page length (some pages may not be fully utilized).
    pub kv_page_last_len: usize,
}

pub fn metadata_store_key(variant: &str) -> String {
    format!("v1.variant.{}.meta", variant)
}

pub fn save_metadata(variant: &str, meta: &VariantMetadata) {
    if let Ok(json) = serde_json::to_string(meta) {
        inferlet::store_set(&metadata_store_key(variant), &json);
    }
}

pub fn load_metadata(variant: &str) -> Option<VariantMetadata> {
    let json = inferlet::store_get(&metadata_store_key(variant))?;
    serde_json::from_str(&json).ok()
}

// ---------------------------------------------------------------------------
// Header parsing — case-insensitive, returns None for "none" or unknown values.
// ---------------------------------------------------------------------------

/// Parse the `X-Hermes-Variant` header from an iterator of (name, value) pairs.
/// Returns `Some(variant)` only when the value matches `is_valid_variant` AND
/// is not "none".
pub fn parse_variant_header<'a>(
    headers_iter: impl Iterator<Item = (&'a str, &'a str)>,
) -> Option<String> {
    for (name, value) in headers_iter {
        if name.eq_ignore_ascii_case("x-hermes-variant") {
            let v = value.trim().to_string();
            if is_valid_variant(&v) && v != "none" {
                return Some(v);
            }
            return None;
        }
    }
    None
}

/// Parses `X-Hermes-Ephemeral` from a header iterator. Returns `None` when the
/// header is absent, empty, or not well-formed base64. Case-insensitive on the
/// header name.
///
/// Phase-1 scope: we only validate that the payload is decodable base64 — we
/// do NOT decode or inspect the content here. Callers receive the raw base64
/// string so that decoded user data never crosses the validation boundary via
/// return value (and, in particular, never gets accidentally logged). Actual
/// platform-specific post-processing is deferred to a later phase.
pub fn parse_ephemeral_header<'a>(
    headers_iter: impl Iterator<Item = (&'a str, &'a str)>,
) -> Option<String> {
    use base64::{Engine as _, engine::general_purpose::STANDARD};
    for (name, value) in headers_iter {
        if name.eq_ignore_ascii_case("x-hermes-ephemeral") {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                return None;
            }
            // Decode attempt is just a well-formedness check; the decoded
            // bytes are discarded so they cannot leak into logs.
            if STANDARD.decode(trimmed).is_ok() {
                return Some(trimmed.to_string());
            }
            return None;
        }
    }
    None
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
    let kv_page_last_len = ctx.get_kv_page_last_len();
    let token_ids = ctx.get_token_ids().to_vec();
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

    // Persist metadata so import path can reconstruct via from_imported_state.
    save_metadata(
        &req.variant_name,
        &VariantMetadata {
            token_ids,
            kv_page_last_len,
        },
    );

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

    // -------------------------------------------------------------------------
    // VariantMetadata key format + serde roundtrip
    // -------------------------------------------------------------------------

    #[test]
    fn metadata_store_key_format() {
        assert_eq!(metadata_store_key("codex"), "v1.variant.codex.meta");
        assert_eq!(metadata_store_key("google"), "v1.variant.google.meta");
        assert_eq!(metadata_store_key("none"), "v1.variant.none.meta");
    }

    #[test]
    fn variant_metadata_json_roundtrip() {
        let meta = VariantMetadata {
            token_ids: vec![1u32, 2, 3, 151644, 77091],
            kv_page_last_len: 42,
        };
        let json = serde_json::to_string(&meta).expect("serialize");
        let back: VariantMetadata = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.token_ids, meta.token_ids);
        assert_eq!(back.kv_page_last_len, meta.kv_page_last_len);
    }

    #[test]
    fn variant_metadata_json_roundtrip_empty_tokens() {
        let meta = VariantMetadata {
            token_ids: vec![],
            kv_page_last_len: 0,
        };
        let json = serde_json::to_string(&meta).expect("serialize");
        let back: VariantMetadata = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.token_ids, meta.token_ids);
        assert_eq!(back.kv_page_last_len, 0);
    }

    // -------------------------------------------------------------------------
    // parse_variant_header
    // -------------------------------------------------------------------------

    #[test]
    fn parse_variant_header_returns_codex() {
        let headers = vec![("x-hermes-variant", "codex")];
        assert_eq!(
            parse_variant_header(headers.iter().map(|(k, v)| (*k, *v))),
            Some("codex".to_string()),
        );
    }

    #[test]
    fn parse_variant_header_returns_google() {
        let headers = vec![("X-Hermes-Variant", "google")];
        assert_eq!(
            parse_variant_header(headers.iter().map(|(k, v)| (*k, *v))),
            Some("google".to_string()),
        );
    }

    #[test]
    fn parse_variant_header_returns_none_for_none_value() {
        let headers = vec![("x-hermes-variant", "none")];
        assert_eq!(
            parse_variant_header(headers.iter().map(|(k, v)| (*k, *v))),
            None,
        );
    }

    #[test]
    fn parse_variant_header_returns_none_for_unknown_value() {
        let headers = vec![("x-hermes-variant", "o1-mini")];
        assert_eq!(
            parse_variant_header(headers.iter().map(|(k, v)| (*k, *v))),
            None,
        );
    }

    #[test]
    fn parse_variant_header_returns_none_when_absent() {
        let headers: Vec<(&str, &str)> = vec![("content-type", "application/json")];
        assert_eq!(
            parse_variant_header(headers.iter().map(|(k, v)| (*k, *v))),
            None,
        );
    }

    #[test]
    fn parse_variant_header_case_insensitive_name() {
        // Header name matching is case-insensitive per HTTP spec
        let headers = vec![("X-HERMES-VARIANT", "codex")];
        assert_eq!(
            parse_variant_header(headers.iter().map(|(k, v)| (*k, *v))),
            Some("codex".to_string()),
        );
    }

    #[test]
    fn parse_variant_header_trims_whitespace() {
        let headers = vec![("x-hermes-variant", " codex ")];
        assert_eq!(
            parse_variant_header(headers.iter().map(|(k, v)| (*k, *v))),
            Some("codex".to_string()),
        );
    }
}

#[cfg(test)]
mod ephemeral_header_tests {
    use super::*;

    fn iter<'a>(items: &'a [(&'a str, &'a str)]) -> impl Iterator<Item = (&'a str, &'a str)> {
        items.iter().copied()
    }

    #[test]
    fn absent_returns_none() {
        assert_eq!(parse_ephemeral_header(iter(&[])), None);
        assert_eq!(
            parse_ephemeral_header(iter(&[("content-type", "application/json")])),
            None,
        );
    }

    #[test]
    fn empty_value_returns_none() {
        assert_eq!(
            parse_ephemeral_header(iter(&[("X-Hermes-Ephemeral", "")])),
            None,
        );
        assert_eq!(
            parse_ephemeral_header(iter(&[("X-Hermes-Ephemeral", "   ")])),
            None,
        );
    }

    #[test]
    fn case_insensitive_name() {
        let b64 = "Y2hhdF9pZD0xMjM="; // "chat_id=123"
        assert_eq!(
            parse_ephemeral_header(iter(&[("x-hermes-ephemeral", b64)])),
            Some(b64.to_string()),
        );
        assert_eq!(
            parse_ephemeral_header(iter(&[("X-HERMES-EPHEMERAL", b64)])),
            Some(b64.to_string()),
        );
    }

    #[test]
    fn rejects_malformed_base64() {
        assert_eq!(
            parse_ephemeral_header(iter(&[("X-Hermes-Ephemeral", "!!!!")])),
            None,
        );
    }
}
