//! OpenAI-compatible chat completions HTTP server inferlet for Pie.
//!
//! ## Endpoints
//!
//! - `POST /v1/chat/completions` - Create a chat completion
//! - `GET  /v1/models`           - List available models
//! - `GET  /health`              - Health check
//!
//! ## Usage
//!
//! ```bash
//! # Build
//! cargo build -p openai-compat --target wasm32-wasip2 --release
//!
//! # Run
//! pie http --path ./openai_compat.wasm --port 8080
//!
//! # Test (non-streaming)
//! curl -X POST http://localhost:8080/v1/chat/completions \
//!   -H "Content-Type: application/json" \
//!   -d '{"model":"auto","messages":[{"role":"user","content":"Hello!"}]}'
//!
//! # Test (streaming)
//! curl -X POST http://localhost:8080/v1/chat/completions \
//!   -H "Content-Type: application/json" \
//!   -d '{"model":"auto","messages":[{"role":"user","content":"Hello!"}],"stream":true}'
//! ```

mod block_cache;
mod constrained_sampler;
mod fork_validate;
mod handler;
mod prompt_cache;
mod prompt_render;
pub(crate) mod session_cache;
pub(crate) mod stop_conditions;
pub(crate) mod tool_format;
mod tool_parser;
mod tools_grammar;
mod types;
mod variant;

use inferlet::sampler::Sample;
use wstd::http::body::IncomingBody;
use wstd::http::server::{Finished, Responder};
use wstd::http::{IntoBody, Method, Request, Response, StatusCode};
use wstd::io::AsyncRead;

#[cfg(target_family = "wasm")]
#[wstd::http_server]
async fn main(mut req: Request<IncomingBody>, res: Responder) -> Finished {
    let path = req.uri().path();
    let method = req.method().clone();

    // Debug: catch any POST to /v1/diag FIRST (before other routes)
    if path == "/v1/diag" {
        let debug_info = format!("method={:?} path={}", method, path);
        let response = Response::builder()
            .header("Content-Type", "text/plain")
            .body(debug_info.into_body())
            .unwrap();
        return res.respond(response).await;
    }

    match (method, path) {
        (Method::POST, "/v1/chat/completions") => {
            let adaptive_stop = req
                .headers()
                .get("x-pie-adaptive-stop")
                .and_then(|v| v.to_str().ok())
                .map(|v| v == "true")
                .unwrap_or(false);
            let n_candidates = req
                .headers()
                .get("x-pie-candidates")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(1)
                .clamp(1, 8);
            let session_id_header = req
                .headers()
                .get("x-pie-session-id")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty());
            // Authorization bearer token, used for auto-deriving a session id
            // when neither X-Pie-Session-Id nor body pie_session is set.
            let auth_bearer = req
                .headers()
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.strip_prefix("Bearer ").or_else(|| s.strip_prefix("bearer ")))
                .map(|s| s.trim().to_string())
                .unwrap_or_default();
            let mut body_bytes = Vec::new();
            if read_body(req.body_mut(), &mut body_bytes).await.is_err() {
                return error_response(res, 400, "Failed to read request body").await;
            }
            handler::handle_chat_completions(
                body_bytes,
                res,
                adaptive_stop,
                n_candidates,
                session_id_header,
                auth_bearer,
            )
            .await
        }

        (Method::GET, "/v1/pie/prompt-cache/capabilities") => {
            let response_json = serde_json::to_string(&prompt_cache::capabilities_response())
                .unwrap_or_else(|_| "{}".to_string());

            let response = Response::builder()
                .header("Content-Type", "application/json")
                .body(response_json.into_body())
                .unwrap();
            res.respond(response).await
        }

        (Method::POST, "/v1/pie/prompt-cache/prefixes:ensure") => {
            let mut body_bytes = Vec::new();
            if read_body(req.body_mut(), &mut body_bytes).await.is_err() {
                return error_response(res, 400, "Failed to read request body").await;
            }
            handler::handle_prefix_ensure(body_bytes, res).await
        }

        (Method::POST, "/v1/pie/prompt-cache/chat-prefix:warm") => {
            let mut body_bytes = Vec::new();
            if read_body(req.body_mut(), &mut body_bytes).await.is_err() {
                return error_response(res, 400, "Failed to read request body").await;
            }
            handler::handle_chat_prefix_warm(body_bytes, res).await
        }

        (Method::POST, "/v1/pie/variant/export") => {
            let mut body_bytes = Vec::new();
            if read_body(req.body_mut(), &mut body_bytes).await.is_err() {
                return error_response(res, 400, "Failed to read request body").await;
            }
            variant::handle_export(body_bytes, res).await
        }

        (Method::GET, "/v1/models") => {
            let models = inferlet::get_all_models();
            let model_list: Vec<serde_json::Value> = models
                .iter()
                .map(|name| {
                    serde_json::json!({
                        "id": name,
                        "object": "model",
                        "owned_by": "pie",
                    })
                })
                .collect();

            let response_json = serde_json::json!({
                "object": "list",
                "data": model_list,
            });

            let response = Response::builder()
                .header("Content-Type", "application/json")
                .body(response_json.to_string().into_body())
                .unwrap();
            res.respond(response).await
        }

        (Method::GET, "/health") => {
            let response = Response::builder()
                .header("Content-Type", "application/json")
                .body(r#"{"status":"ok"}"#.into_body())
                .unwrap();
            res.respond(response).await
        }

        (Method::GET, "/v1/debug/store") => {
            let keys = ["v2.debug.session_match", "v2.session_state", "v2.active_session_id", "v2.prefix_checkpoint"];
            let entries: Vec<serde_json::Value> = keys.iter().map(|k| {
                serde_json::json!({
                    "key": k,
                    "value": inferlet::store_get(k),
                })
            }).collect();
            let json = serde_json::json!({"store": entries});
            let response = Response::builder()
                .header("Content-Type", "application/json")
                .body(json.to_string().into_body())
                .unwrap();
            res.respond(response).await
        }

        (Method::GET, "/v1/debug/kv-exports") => {
            use inferlet::forward::Forward;
            let model = inferlet::get_auto_model();
            let queue = model.create_queue();
            let exports = queue.get_all_exported_kv_pages();
            let json = serde_json::json!({
                "exports": exports.iter().map(|(name, count)| {
                    serde_json::json!({"name": name, "page_count": count})
                }).collect::<Vec<_>>(),
                "total": exports.len(),
            });
            let response = Response::builder()
                .header("Content-Type", "application/json")
                .body(json.to_string().into_body())
                .unwrap();
            res.respond(response).await
        }

        // GET debug/grammar: test constrained sampler construction and initial mask
        (Method::GET, "/v1/debug/grammar") => {
            use std::time::Instant;
            let mut info = Vec::new();
            let t0 = Instant::now();

            let model = inferlet::get_auto_model();
            let tokenizer = model.get_tokenizer();
            info.push(format!("model: {}", model.get_name()));

            // Detect tool call format from model's chat template
            let t_detect = Instant::now();
            let detected_format = crate::tool_format::ToolCallFormat::detect(&model).await;
            let detect_ms = t_detect.elapsed().as_millis();

            match &detected_format {
                Some(fmt) => {
                    info.push(format!("tool_format detected in {}ms:", detect_ms));
                    info.push(format!("  open_text: {:?}  open_token_ids: {:?}", fmt.open_text, fmt.open_token_ids));
                    info.push(format!("  close_text: {:?}  close_token_ids: {:?}", fmt.close_text, fmt.close_token_ids));
                    info.push(format!("  single_special_tokens: {}", fmt.single_special_tokens));
                    info.push(format!("  name_key: {:?}  args_key: {:?}", fmt.name_key, fmt.args_key));
                }
                None => {
                    info.push(format!("tool_format: NOT DETECTED ({}ms)", detect_ms));
                }
            }

            // Build a simple test grammar using detected format
            let tools = vec![crate::types::Tool {
                tool_type: "function".to_string(),
                function: crate::types::FunctionDef {
                    name: "search".to_string(),
                    description: Some("Search".to_string()),
                    parameters: Some(serde_json::json!({"type":"object","properties":{"q":{"type":"string"}}})),
                },
            }];

            let grammar = if let Some(fmt) = &detected_format {
                crate::tools_grammar::tools_to_lark_grammar(&tools, fmt, &crate::types::ToolChoice::Auto)
            } else {
                "(no format detected — cannot build grammar)".to_string()
            };
            info.push(format!("grammar ({} chars): {}", grammar.len(), grammar.replace('\n', " | ")));

            // Get tokenizer info
            let vocabs = tokenizer.get_vocabs();
            info.push(format!("vocab: {} ids, {} words", vocabs.0.len(), vocabs.1.len()));
            let special = tokenizer.get_special_tokens();
            info.push(format!("special_tokens: {} ids", special.0.len()));
            let split_regex = tokenizer.get_split_regex();
            info.push(format!("split_regex ({} chars): {}...", split_regex.len(), &split_regex[..split_regex.len().min(100)]));

            // Check if special tokens include tool-related names
            let spec_names: Vec<String> = special.1.iter()
                .filter_map(|b| String::from_utf8(b.clone()).ok())
                .filter(|s| s.contains("tool") || s.contains("TOOL") || s.contains("python_tag"))
                .collect();
            info.push(format!("tool-related special tokens: {:?}", spec_names));

            // Find EOS
            let eos_tokens = model.eos_tokens();
            let eos_id = eos_tokens.iter().find(|t| t.len() == 1).and_then(|t| t.first().copied());
            info.push(format!("eos_tokens: {:?}, single_eos_id: {:?}", eos_tokens, eos_id));

            // Try building the sampler (only if format was detected)
            if let Some(fmt) = &detected_format {
                let t_build = Instant::now();
                let build_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    crate::constrained_sampler::ConstrainedSampler::new(
                        tokenizer.get_vocabs(),
                        tokenizer.get_special_tokens(),
                        tokenizer.get_split_regex(),
                        grammar.clone(),
                        eos_id.unwrap_or(0),
                        true, // escape_non_printable
                    )
                }));
                let build_ms = t_build.elapsed().as_millis();
                match build_result {
                    Ok(sampler) => {
                        info.push(format!("sampler build: OK in {}ms", build_ms));

                        // Test sampling with open marker token + text tokens
                        let hello_ids = tokenizer.tokenize("Hello world the");
                        let mut mixed_ids = hello_ids.clone();
                        mixed_ids.extend_from_slice(&fmt.open_token_ids);
                        if !fmt.close_token_ids.is_empty() {
                            mixed_ids.extend_from_slice(&fmt.close_token_ids);
                        }
                        let mixed_probs: Vec<f32> = vec![0.05; mixed_ids.len()];
                        let sampler_fresh = crate::constrained_sampler::ConstrainedSampler::new(
                            tokenizer.get_vocabs(),
                            tokenizer.get_special_tokens(),
                            tokenizer.get_split_regex(),
                            grammar.clone(),
                            eos_id.unwrap_or(0),
                            true,
                        );
                        let token = sampler_fresh.sample(&mixed_ids, &mixed_probs);
                        info.push(format!("sample(hello+marker_ids={:?}): token={} ({})",
                            mixed_ids, token, tokenizer.detokenize(&[token])));

                        let _ = sampler; // consumed
                    }
                    Err(e) => {
                        info.push(format!("sampler build: PANIC in {}ms: {:?}", build_ms, e.downcast_ref::<String>()));
                    }
                }
            }

            info.push(format!("total elapsed: {}ms", t0.elapsed().as_millis()));

            let json = serde_json::json!({"debug": info});
            let response = Response::builder()
                .header("Content-Type", "application/json")
                .body(json.to_string().into_body())
                .unwrap();
            res.respond(response).await
        }

        // POST diag: test each step to find where the crash happens
        (Method::POST, "/v1/diag") => {
            let mut steps = Vec::new();
            steps.push("1:entry");

            // Step 2: Read body
            let mut body_bytes = Vec::new();
            if read_body(req.body_mut(), &mut body_bytes).await.is_err() {
                steps.push("2:body_FAIL");
            } else {
                steps.push("2:body_ok");
            }

            // Step 3: Get model
            let model = inferlet::get_auto_model();
            steps.push("3:model_ok");

            // Step 4: Create context
            let mut ctx = model.create_context();
            steps.push("4:ctx_ok");

            // Step 5: Fill
            ctx.fill("Hello");
            steps.push("5:fill_ok");

            // Step 6: Generate (1 token only)
            let sampler = inferlet::Sampler::greedy();
            let stop = inferlet::stop_condition::max_len(1);
            let out = ctx.generate(sampler, stop).await;
            steps.push("6:generate_ok");

            let body = format!("{}\noutput: {}", steps.join("\n"), out);
            let response = Response::builder()
                .header("Content-Type", "text/plain")
                .body(body.into_body())
                .unwrap();
            res.respond(response).await
        }

        (Method::OPTIONS, _) => {
            let response = Response::builder()
                .header("Access-Control-Allow-Origin", "*")
                .header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
                .header(
                    "Access-Control-Allow-Headers",
                    "Content-Type, Authorization",
                )
                .body("".into_body())
                .unwrap();
            res.respond(response).await
        }

        _ => not_found(res).await,
    }
}

/// Read the entire request body into a Vec<u8>.
async fn read_body(body: &mut IncomingBody, buf: &mut Vec<u8>) -> Result<(), ()> {
    let mut chunk = [0u8; 4096];
    loop {
        match body.read(&mut chunk).await {
            Ok(0) => break,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
            Err(_) => return Err(()),
        }
    }
    Ok(())
}

/// Return an error response with JSON body.
pub async fn error_response(res: Responder, status: u16, message: &str) -> Finished {
    let error = serde_json::json!({
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "param": null,
            "code": null,
        }
    });

    let response = Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(error.to_string().into_body())
        .unwrap();

    res.respond(response).await
}

/// Return 404 Not Found.
async fn not_found(res: Responder) -> Finished {
    let error = serde_json::json!({
        "error": {
            "message": "Endpoint not found",
            "type": "not_found",
            "param": null,
            "code": null,
        }
    });

    let response = Response::builder()
        .status(StatusCode::NOT_FOUND)
        .header("Content-Type", "application/json")
        .body(error.to_string().into_body())
        .unwrap();

    res.respond(response).await
}
