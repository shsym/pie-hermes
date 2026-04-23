//! Request handler for POST /v1/chat/completions endpoint.

use crate::prompt_cache;
use crate::stop_conditions::{
    DegenerateLoopDetector, DynStopCondition, ToolCallBudget, ToolCallComplete,
};
use crate::types::*;
use inferlet::Sampler;
use inferlet::forward::Forward;
use inferlet::stop_condition::{StopCondition, ends_with_any, max_len};
use sha2::{Digest, Sha256};
use wstd::http::body::BodyForthcoming;
use wstd::http::server::{Finished, Responder};
use wstd::http::{IntoBody, Response};
use wstd::io::AsyncWrite;

/// Derive a stable session id from the Authorization bearer token and the first
/// system message in the request. The bearer is a mandatory tenant identifier:
/// without it we cannot distinguish "alice using system prompt X" from "bob
/// using system prompt X" and would bucket their KV together.
///
/// Returns None when the bearer is empty. Anonymous callers bypass the
/// session-affine path and run one-shot.
fn auto_session_id(bearer: &str, request: &ChatCompletionRequest) -> Option<String> {
    if bearer.is_empty() {
        return None;
    }
    let system = request
        .messages
        .iter()
        .find(|m| m.role == "system")
        .map(|m| m.content.as_str())
        .unwrap_or("");
    // Take the first 500 *characters*, not bytes — slicing by bytes at a
    // multibyte boundary panics the whole inferlet.
    let system_head: String = system.chars().take(500).collect();
    let mut hasher = Sha256::new();
    hasher.update(bearer.as_bytes());
    hasher.update(b":");
    hasher.update(system_head.as_bytes());
    let digest = hasher.finalize();
    let hex: String = digest.iter().map(|b| format!("{:02x}", b)).collect();
    Some(format!("auto-{}", &hex[..16]))
}

/// Scope a caller-supplied `X-Pie-Session-Id` value with the bearer token so
/// two tenants sending the same raw header value don't land in the same KV
/// bucket. Without this, any client could read or poison another tenant's
/// session by guessing or leaking their id.
///
/// Returns None when the bearer is empty — an unauthenticated client cannot
/// be granted session-scoped writes.
fn scoped_session_id(bearer: &str, raw: &str) -> Option<String> {
    if bearer.is_empty() || raw.is_empty() {
        return None;
    }
    let mut hasher = Sha256::new();
    hasher.update(bearer.as_bytes());
    hasher.update(b":");
    hasher.update(raw.as_bytes());
    let digest = hasher.finalize();
    let hex: String = digest.iter().map(|b| format!("{:02x}", b)).collect();
    Some(format!("scoped-{}", &hex[..16]))
}

pub async fn handle_prefix_ensure(body_bytes: Vec<u8>, responder: Responder) -> Finished {
    let request: PrefixEnsureRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => {
            return crate::error_response(responder, 400, &format!("Invalid JSON: {}", e)).await;
        }
    };

    let model = inferlet::get_auto_model();
    let response = match prompt_cache::ensure_prefix(&model, request).await {
        Ok(response) => response,
        Err(message) => {
            return crate::error_response(responder, 400, &message).await;
        }
    };

    let json = serde_json::to_string(&response).unwrap_or_default();
    let http_response = Response::builder()
        .header("Content-Type", "application/json")
        .body(json.into_body())
        .unwrap();

    responder.respond(http_response).await
}

/// POST /v1/pie/prompt-cache/chat-prefix:warm
///
/// Operator-triggered warmup: takes a subset of ChatCompletionRequest
/// ({messages, tools, chat_template_kwargs}), runs the SAME reorder +
/// format_chat_tokens path the real chat handler uses, prefills a
/// standalone context, and populates the block-level cache so the first
/// real user request hits Level 2 immediately.  No response stream, no
/// decode — just prefill + export.
pub async fn handle_chat_prefix_warm(body_bytes: Vec<u8>, responder: Responder) -> Finished {
    let request: ChatCompletionRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => {
            return crate::error_response(responder, 400, &format!("Invalid JSON: {}", e)).await;
        }
    };

    let model = inferlet::get_auto_model();

    // Reorder exactly the way prepare_session_execution does so the
    // resulting block chain hashes line up with real chat requests.
    let reordered = crate::prompt_render::reorder_system_sections(&request.messages);
    let tokens = match crate::prompt_render::format_chat_tokens(
        &model,
        &reordered,
        &request.tools,
        &request.chat_template_kwargs,
        false, // no generation prompt for the warmup
    )
    .await
    {
        Ok(t) if !t.is_empty() => t,
        Ok(_) => {
            return crate::error_response(responder, 400, "warmup tokens empty").await;
        }
        Err(e) => {
            return crate::error_response(responder, 400, &format!("tokenize: {}", e)).await;
        }
    };

    let page_size = model.get_kv_page_size() as usize;
    if page_size == 0 || tokens.len() < page_size {
        return crate::error_response(
            responder,
            400,
            "warmup prompt shorter than one KV page",
        )
        .await;
    }

    // Prefill a standalone ctx.  This ctx never serves a user response;
    // it exists solely to populate the block cache export registry.
    let mut ctx = model.create_context();
    ctx.fill_tokens(tokens.clone());
    ctx.flush().await;

    let saved = crate::block_cache::save_ctx_blocks(&ctx, &tokens, page_size);

    let resp_json = serde_json::json!({
        "status": if saved { "warmed" } else { "noop (already cached)" },
        "tokens_prefilled": tokens.len(),
        "blocks": tokens.len() / page_size,
        "page_size": page_size,
    });
    let http_response = Response::builder()
        .header("Content-Type", "application/json")
        .body(serde_json::to_string(&resp_json).unwrap_or_default().into_body())
        .unwrap();
    responder.respond(http_response).await
}

/// Handle POST /v1/chat/completions.
pub async fn handle_chat_completions(
    body_bytes: Vec<u8>,
    responder: Responder,
    adaptive_stop: bool,
    n_candidates: usize,
    session_id_header: Option<String>,
    auth_bearer: String,
    variant_header: Option<String>,
    ephemeral_header: Option<String>,
) -> Finished {
    let t_entry = std::time::Instant::now();
    let body_len = body_bytes.len();

    // Phase-2.0 ephemeral re-prefill: when X-Hermes-Ephemeral is present, decode
    // the base64 payload now so we can inject it as a system message into
    // request.messages BEFORE prepare_execution renders + tokenizes. The prefix
    // cache hit on the leading system region is preserved because the injected
    // content lands AFTER the cached system prefix; the chat template adds the
    // appropriate role markers (we do NOT call ctx.fill_tokens after prepare_*,
    // which would corrupt the trailing `<|im_start|>assistant\n` generation
    // prompt).
    //
    // Logging discipline (VENDOR_SOURCE.md #5): we are in the sync prefix of an
    // async fn — NO eprintln!/println! before the first .await. The decoded
    // payload itself must NEVER be logged (PII boundary); a length-only log is
    // OK only after the first .await fires.
    let ephemeral_decoded: Option<String> = ephemeral_header
        .as_deref()
        .and_then(crate::variant::decode_ephemeral_payload);

    // Defense-in-depth: parse_ephemeral_header (variant.rs:85+) already validates
    // well-formedness, so a Some(header) that decodes to None should be unreachable.
    // Catches if parse_ephemeral_header is ever relaxed without updating the decode
    // path. No-op in release.
    debug_assert!(
        ephemeral_header.is_none() || ephemeral_decoded.is_some(),
        "ephemeral header passed parse_ephemeral_header validation but decode_ephemeral_payload returned None — invariant broken"
    );

    // Parse request
    let mut request: ChatCompletionRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => {
            return crate::error_response(responder, 400, &format!("Invalid JSON: {}", e)).await;
        }
    };
    let t_json = t_entry.elapsed();

    // Inject the ephemeral payload as a system message at the END of the
    // existing system block (right after the last system-role message, before
    // the first non-system message). If there are no system messages at all,
    // prepend at index 0. Routing through request.messages lets the chat
    // template emit the proper role markers and stop-token discipline.
    if let Some(text) = ephemeral_decoded.as_ref() {
        let insert_at = request
            .messages
            .iter()
            .position(|m| m.role != "system")
            .unwrap_or(request.messages.len());
        request.messages.insert(
            insert_at,
            ChatMessage {
                role: "system".to_string(),
                content: format!("[ephemeral metadata]\n{}", text),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
        );
    }

    // Auto-enable adaptive stop conditions when the request advertises tools.
    // The `x-pie-adaptive-stop` header is set by pie-native clients but not by
    // generic OpenAI clients (OpenClaw, LangChain, …). Without adaptive_stop
    // the three defensive bounds — ToolCallBudget(3), ToolCallComplete,
    // DegenerateLoopDetector — are all inactive, so a looping model can emit
    // dozens of redundant tool_call blocks in a single turn until it hits
    // max_tokens (observed in task #19 v6: 57 calls each to the same two
    // integration ids). Tools ⇒ turn into budgeted turn.
    let has_tools = request
        .tools
        .as_ref()
        .is_some_and(|tools| !tools.is_empty());
    let adaptive_stop = adaptive_stop || has_tools;

    // Resolve session_id in priority order:
    //   1. body.pie_session.session_id (explicit wire contract; unchanged for
    //      existing callers that pass through replay-benchmark-v2 etc.)
    //   2. X-Pie-Session-Id header, scoped by the bearer token so two tenants
    //      can't land in the same KV bucket by sending the same raw id.
    //   3. auto-derive from bearer + first system message.
    //
    // Tiers 2 and 3 both require a non-empty bearer token; unauthenticated
    // callers fall through to the non-session path rather than share a bucket
    // keyed only on a guessable system prompt.
    //
    // Phase-1 scope: when X-Hermes-Variant is set, skip session-id derivation
    // entirely so the request uses the non-session path.  Coexisting variant +
    // session KV semantics on turn 2+ require storing variant tokens in session
    // state — deferred to a later phase.
    if request.pie_session.is_none() && variant_header.is_none() {
        let session_id = session_id_header
            .as_deref()
            .and_then(|raw| scoped_session_id(&auth_bearer, raw))
            .or_else(|| auto_session_id(&auth_bearer, &request));
        if let Some(session_id) = session_id {
            request.pie_session = Some(crate::types::PieSessionRequest { session_id });
        }
    }

    if request.messages.is_empty() {
        return crate::error_response(responder, 400, "No messages provided").await;
    }

    let max_tokens = request.max_tokens.unwrap_or(4096);
    // Match vLLM's OpenAI-server defaults so OpenClaw / OpenAI clients that
    // omit sampling params get the same distribution on both backends.
    // Previously 0.6 / 0.95 (Qwen-style "balanced" defaults), but the bench
    // showed pie looping 30-50x more than vllm on list-iterate tasks — the
    // sampling drift was a confounder. vLLM ChatCompletionRequest defaults
    // are temperature=0.7, top_p=1.0.
    let temperature = request.temperature.unwrap_or(0.7);
    let top_p = request.top_p.unwrap_or(1.0);

    let mut prepared = match prepare_execution(&request, variant_header.as_deref()).await {
        Ok(prepared) => prepared,
        Err(message) => {
            return crate::error_response(responder, 400, &message).await;
        }
    };
    let t_prepare = t_entry.elapsed();

    // Phase-2.0: record an approximate token count of the injected ephemeral
    // payload. Approximation only — `tokenize(text)` does NOT include the role
    // markers added by the chat template, so actual prompt-token growth is
    // slightly higher. The model is bound after prepare_execution; we tokenize
    // here purely to populate telemetry. Logging the LENGTH is OK (we are
    // post-await), logging the text itself is NOT.
    if let Some(text) = ephemeral_decoded.as_ref() {
        let approx = prepared.model.get_tokenizer().tokenize(text).len() as u32;
        if let Some(cache) = prepared.pie_cache.as_mut() {
            cache.ephemeral_tokens_appended = Some(approx);
        }
    }

    if request.stream {
        return handle_streaming(responder, prepared, &request, max_tokens, temperature, top_p, t_entry, t_json, t_prepare, body_len, adaptive_stop).await;
    }

    // --- Non-streaming path ---
    let PreparedExecution {
        model,
        mut ctx,
        prompt_tokens,
        pie_cache,
        profile: _,
        session_incoming_tokens,
    } = prepared;
    let model_name = model.get_name();

    // Fork-validate path: generate N candidates and pick the best.
    let (generated, fork_meta) = if n_candidates > 1 {
        // Flush pending fills so forked contexts share committed KV state.
        ctx.flush().await;
        let result = crate::fork_validate::generate_best_of_n(
            &ctx, &model, n_candidates, max_tokens, temperature, top_p, adaptive_stop,
        ).await;
        let meta = Some((result.candidate_index, result.n_candidates, result.total_generated_tokens));
        (result.text, meta)
    } else {
        // Single-candidate path (original behavior).
        let sampler = build_sampler(&model, &request, temperature, top_p).await;
        let base_cond = max_len(max_tokens).or(ends_with_any(model.eos_tokens()));
        let stop_cond = build_stop_condition(base_cond, adaptive_stop, &model);
        let generated = ctx.generate(sampler, stop_cond).await;
        let stop_strings = eos_token_strings(&model);
        let generated = strip_stop_tokens(&generated, &stop_strings);
        (generated, None)
    };

    // Skip session KV save for fork path — forked contexts are ephemeral.
    // Future: export winning fork's KV.
    if fork_meta.is_none() {
        // block_cache::lookup_longest_prefix is permanently disabled (see module
        // docstring); it only writes entries that nothing ever reads back.  Until
        // the lookup side is restored, skip block_cache save entirely so session
        // KV retention actually persists across turns.  The old short-circuit
        // ("if block save did anything, skip session_kv") left session state
        // unwritten whenever block_cache ran — breaking multi-turn KV reuse.
        save_session_kv_state(&ctx, &request, session_incoming_tokens.as_deref());
    }

    // Parse tool calls from model output
    let (content, tool_calls) = crate::tool_parser::parse_tool_calls(&generated);
    let has_tool_calls = !tool_calls.is_empty();

    let completion_tokens = model.get_tokenizer().tokenize(&generated).len() as u32;

    let response = ChatCompletionResponse {
        id: generate_id("chatcmpl-"),
        object: "chat.completion".to_string(),
        created: 0,
        model: model_name,
        choices: vec![Choice {
            index: 0,
            message: ChoiceMessage {
                role: "assistant".to_string(),
                content,
                tool_calls: if has_tool_calls {
                    Some(tool_calls)
                } else {
                    None
                },
            },
            finish_reason: if has_tool_calls {
                "tool_calls".to_string()
            } else {
                "stop".to_string()
            },
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
        pie_cache,
    };

    let json = serde_json::to_string(&response).unwrap_or_default();

    let mut builder = Response::builder()
        .header("Content-Type", "application/json");
    if let Some((idx, total, total_tokens)) = fork_meta {
        builder = builder
            .header("x-pie-candidate-index", idx.to_string())
            .header("x-pie-candidates-generated", total.to_string())
            .header("x-pie-fork-total-tokens", total_tokens.to_string());
    }
    let http_response = builder
        .body(json.into_body())
        .unwrap();

    responder.respond(http_response).await
}

/// Build a stop condition, optionally composing adaptive stop conditions for agentic workloads.
///
/// When `adaptive_stop` is true, adds (all OR'd with base):
/// - ToolCallBudget: caps total tool calls per turn (primary tool-call limiter)
/// - ToolCallComplete: stops trailing text after tool calls (catches post-tool rambling)
/// - DegenerateLoopDetector: catches infinite repetition loops
fn build_stop_condition<S: StopCondition + 'static>(
    base: S,
    adaptive_stop: bool,
    model: &inferlet::Model,
) -> DynStopCondition {
    let mut cond = DynStopCondition::new(base);
    if adaptive_stop {
        let tokenizer = model.get_tokenizer();
        cond = cond
            // Budget is the primary tool-call limiter — stops after N complete calls.
            .or_dyn(ToolCallBudget::new(&tokenizer, 3))
            // Catches trailing garbage after tool calls (>20 non-tag tokens after last </tool_call>).
            .or_dyn(ToolCallComplete::new(&tokenizer))
            // Safety net for degenerate repetition (10-gram repeated 3x).
            .or_dyn(DegenerateLoopDetector::default());
    }
    cond
}

/// Build a sampler, using constrained decoding when tools are present
/// and `tool_choice` is not `None`.
///
/// When the request includes `tools`, attempts to build a grammar-constrained
/// sampler via llguidance that enforces valid tool-call JSON inside
/// model-specific tool-call markers. Falls back to standard top-p sampling
/// if constrained sampler construction fails.
async fn build_sampler(
    model: &inferlet::Model,
    request: &ChatCompletionRequest,
    temperature: f32,
    top_p: f32,
) -> Sampler {
    // tool_choice=none → skip constrained decoding entirely
    if matches!(request.tool_choice, ToolChoice::None) {
        return Sampler::top_p(temperature, top_p);
    }

    if let Some(tools) = &request.tools {
        if !tools.is_empty() {
            // When tools are present, default to Qwen2.5's
            // generation_config.json values (rep_penalty=1.05, top_k=20,
            // top_p=0.8) to match what vLLM auto-injects at load. Without
            // top_k/top_p, the stochastic draw over the full
            // grammar-allowed distribution can hit rare tokens and
            // hallucinate (Phase D1 evidence: turn-0 bogus "1234567890"
            // integration ID). Clients can override per-request.
            let rep_penalty = request
                .repetition_penalty
                .filter(|p| p.is_finite() && *p > 0.0)
                .unwrap_or(1.05);
            let sampler_top_p = request
                .top_p
                .filter(|p| p.is_finite() && *p > 0.0 && *p < 1.0)
                .unwrap_or(0.8);
            let sampler_top_k = request
                .top_k
                .filter(|k| *k > 0)
                .unwrap_or(20);
            match build_tool_call_sampler(
                model,
                tools,
                &request.tool_choice,
                temperature,
                rep_penalty,
                sampler_top_p,
                sampler_top_k,
            )
            .await
            {
                Ok(s) => return s,
                Err(e) => {
                    eprintln!("[WARN] Failed to build constrained sampler: {e}. Falling back to top_p.");
                }
            }
        }
    }
    Sampler::top_p(temperature, top_p)
}

/// Build a grammar-constrained sampler for tool-call generation.
///
/// Detects the model's tool call format (open/close markers, JSON keys) by
/// probing the chat template, then builds a grammar that matches.
async fn build_tool_call_sampler(
    model: &inferlet::Model,
    tools: &[Tool],
    tool_choice: &crate::types::ToolChoice,
    temperature: f32,
    rep_penalty: f32,
    top_p: f32,
    top_k: u32,
) -> Result<Sampler, String> {
    let tokenizer = model.get_tokenizer();

    // Detect tool call format from the model's chat template + special tokens
    let format = crate::tool_format::ToolCallFormat::detect(model)
        .await
        .ok_or("Could not detect tool call format for this model")?;

    let grammar = crate::tools_grammar::tools_to_lark_grammar(tools, &format, tool_choice);

    // Find EOS token
    let eos_id = model
        .eos_tokens()
        .iter()
        .find(|t| t.len() == 1)
        .and_then(|t| t.first().copied())
        .ok_or("No single EOS token found")?;

    // Determine if we need non-printable escaping (Qwen/DeepSeek models)
    let name_lower = model.get_name().to_lowercase();
    let escape_non_printable =
        name_lower.starts_with("qwen") || name_lower.starts_with("deepseek");

    let sampler = crate::constrained_sampler::ConstrainedSampler::with_options(
        tokenizer.get_vocabs(),
        tokenizer.get_special_tokens(),
        tokenizer.get_split_regex(),
        grammar,
        eos_id,
        escape_non_printable,
        crate::constrained_sampler::SamplerOptions {
            rep_penalty,
            top_p: Some(top_p),
            top_k: Some(top_k),
            seed: None,
        },
    );

    Ok(Sampler::Custom {
        temperature,
        sampler: Box::new(sampler),
        // Penalties are applied upstream by the constrained sampler itself
        // (see constrained_sampler.rs); pass Default here so the SDK layer
        // doesn't re-apply them. This field became mandatory in the pie SDK
        // at 83d1a6ac (unified sampling: gen_config + Penalties — task23).
        penalties: inferlet::sampler::Penalties::default(),
    })
}

/// Get EOS token strings by detokenizing the model's EOS token IDs,
/// plus well-known special token markers that may not round-trip through tokenization.
pub(crate) fn eos_token_strings(model: &inferlet::Model) -> Vec<String> {
    let tokenizer = model.get_tokenizer();
    let mut strings: Vec<String> = model
        .eos_tokens()
        .into_iter()
        .map(|ids| tokenizer.detokenize(&ids))
        .filter(|s| !s.is_empty())
        .collect();

    // Add well-known special token text that may not round-trip through tokenize/detokenize
    for s in [
        "<|im_end|>",
        "<|endoftext|>",
        "<|im_start|>",
        "</s>",
        "<|eot_id|>",
    ] {
        if !strings.iter().any(|existing| existing == s) {
            strings.push(s.to_string());
        }
    }
    strings
}

/// Strip any trailing stop token strings from generated text.
pub(crate) fn strip_stop_tokens(text: &str, stop_tokens: &[String]) -> String {
    let mut result = text.to_string();
    for token in stop_tokens {
        if let Some(stripped) = result.strip_suffix(token.as_str()) {
            result = stripped.to_string();
        }
    }
    result
}

/// Fill context using server-side HF template rendering via format_chat WIT binding.
async fn fill_context(
    model: &inferlet::Model,
    ctx: &mut inferlet::Context,
    request: &ChatCompletionRequest,
) -> usize {
    let token_ids = crate::prompt_render::format_chat_tokens(
        model,
        &request.messages,
        &request.tools,
        &request.chat_template_kwargs,
        true,
    )
    .await
    .unwrap_or_default();
    let prompt_tokens = token_ids.len();
    ctx.fill_tokens(token_ids);
    prompt_tokens
}

/// Compute the next text delta to emit, given a stream of generated token ids.
///
/// Mirrors vLLM's `detokenize_incrementally`. The two offsets track positions in
/// the token vector:
/// * `prefix_offset` — start of the rolling window passed to the detokenizer
///   for the *previous* emit (gives the BPE merger enough context so the new
///   bytes join the right grapheme).
/// * `read_offset`  — first token whose text has not yet been emitted.
///
/// Per call we detokenize the same window twice (once stopping at
/// `read_offset`, once including all new tokens up to `token_ids.len()`) and
/// emit the byte-suffix that just appeared. If the suffix ends on U+FFFD we
/// hold the emit back until subsequent tokens complete the codepoint —
/// otherwise multi-byte characters (Chinese, emoji, …) leak as `?` over SSE.
///
/// Returns `(delta, new_prefix_offset, new_read_offset)`. When the delta is
/// empty the offsets are returned unchanged so the caller may try again on the
/// next token.
pub(crate) fn incremental_detokenize_step<F>(
    detokenize: F,
    token_ids: &[u32],
    prefix_offset: usize,
    read_offset: usize,
) -> (String, usize, usize)
where
    F: Fn(&[u32]) -> String,
{
    const REPLACEMENT_CHAR: char = '\u{FFFD}';
    let n = token_ids.len();
    if n <= read_offset {
        return (String::new(), prefix_offset, read_offset);
    }
    let prefix_text = if read_offset > prefix_offset {
        detokenize(&token_ids[prefix_offset..read_offset])
    } else {
        String::new()
    };
    let new_text_full = detokenize(&token_ids[prefix_offset..n]);
    if new_text_full.len() > prefix_text.len()
        && !new_text_full.ends_with(REPLACEMENT_CHAR)
    {
        let delta = new_text_full[prefix_text.len()..].to_string();
        (delta, read_offset, n)
    } else {
        (String::new(), prefix_offset, read_offset)
    }
}

/// Handle streaming SSE response with per-step timing.
async fn handle_streaming(
    responder: Responder,
    prepared: PreparedExecution,
    request: &ChatCompletionRequest,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    t_entry: std::time::Instant,
    t_json: std::time::Duration,
    t_prepare: std::time::Duration,
    body_len: usize,
    adaptive_stop: bool,
) -> Finished {
    let PreparedExecution {
        model,
        mut ctx,
        prompt_tokens: _,
        pie_cache,
        profile: prepare_profile,
        session_incoming_tokens,
    } = prepared;
    let model_name = model.get_name();
    let tokenizer = model.get_tokenizer();

    let sampler = build_sampler(&model, request, temperature, top_p).await;
    // Custom (constrained) samplers are incompatible with multi-step decode_n
    // (engine-side sampling). Fall back to single-step decode when constrained.
    let is_constrained = matches!(sampler, Sampler::Custom { .. });
    let base_cond = max_len(max_tokens).or(ends_with_any(model.eos_tokens()));
    let stop_cond = build_stop_condition(base_cond, adaptive_stop, &model);
    let stop_strings = eos_token_strings(&model);
    let chunk_id = generate_id("chatcmpl-");

    // Start SSE response with BodyForthcoming for true streaming
    let sse_response = Response::builder()
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .body(BodyForthcoming)
        .unwrap();

    let mut body = responder.start_response(sse_response);

    // Helper macro: write + flush an SSE data line
    macro_rules! emit {
        ($data:expr) => {{
            let line = format!("data: {}\n\n", $data);
            if body.write_all(line.as_bytes()).await.is_err() {
                return Finished::finish(body, Ok(()), None);
            }
            if body.flush().await.is_err() {
                return Finished::finish(body, Ok(()), None);
            }
        }};
    }

    // First chunk: role announcement
    let first_chunk = ChatCompletionChunk {
        id: chunk_id.clone(),
        object: "chat.completion.chunk".to_string(),
        created: 0,
        model: model_name.clone(),
        pie_cache: None,
        choices: vec![ChunkChoice {
            index: 0,
            delta: ChunkDelta {
                role: Some("assistant".to_string()),
                content: None,
                tool_calls: None,
            },
            finish_reason: None,
        }],
    };
    emit!(serde_json::to_string(&first_chunk).unwrap_or_default());

    // Streaming decode: single decode_n call with large N.
    // The runtime streams tokens via mpsc internally (no yield-boundary stall).
    // decode_n returns all tokens at once after generation completes.
    // Then we process and emit SSE chunks.
    //
    // This eliminates the c>=2 stall (~32ms per yield boundary) at the cost of
    // slightly delayed SSE streaming (tokens arrive in burst after generation).
    // For true per-token SSE streaming, a future change would use decode_stream
    // with an async callback.
    // Multi-step decode with per-token streaming runtime support.
    // decode_n(8) still yields every 8 tokens for stop checking + SSE emission.
    // The runtime streams tokens via mpsc internally (no accumulation delay),
    // but the yield-boundary stall remains until EOS detection is added to
    // the runtime's continuation loop.
    // Custom (constrained) samplers must use single-step decode because
    // multi-step decode_n uses engine-side sampling which bypasses the sampler.
    //
    // Hybrid decode_n schedule:
    //   - first yield:  dn=1  → first token emitted after 1 decode step (min TTFT)
    //   - subsequent:   dn=8  → amortize wasmtime async-yield overhead over 8 tokens
    //
    // Measured on task 88477 (2026-04-11, A40, Qwen2.5-7B, D3 registered_prefix):
    //   dn=8 static:  TTFT 601 ms, total 1661 ms, 26.6 ms/tok
    //   dn=1 static:  TTFT 458 ms, total 1763 ms, 32.6 ms/tok
    // Hybrid keeps the TTFT win of dn=1 without the total-time regression of dn=1.
    //
    // Custom (constrained) samplers must use single-step decode because
    // multi-step decode_n uses engine-side sampling which bypasses the sampler.
    let steady_decode_n: u32 = if is_constrained { 1 } else { 8 };
    let first_decode_n: u32 = 1;

    let mut generated_token_ids = Vec::new();
    let mut full_text = String::new();
    let mut in_tool_call = false;
    // vLLM-style incremental detokenization state — see
    // `incremental_detokenize_step` for the algorithm.
    let mut detok_prefix_offset: usize = 0;
    let mut detok_read_offset: usize = 0;
    let mut t_decode_total = std::time::Duration::ZERO;
    let mut t_detok_total = std::time::Duration::ZERO;
    let mut t_serial_total = std::time::Duration::ZERO;
    let mut t_write_total = std::time::Duration::ZERO;
    let mut t_flush_total = std::time::Duration::ZERO;
    let mut n_steps: u32 = 0;
    let mut n_yields: u32 = 0;
    let mut n_emits: u32 = 0;
    let t_loop_start = std::time::Instant::now();
    let t_entry_to_loop = t_entry.elapsed();

    // Emit pre-decode timing as SSE comment
    {
        let trace = format!(
            ": [INFERLET-TRACE] json={:.1}ms prepare={:.1}ms entry_to_loop={:.1}ms body_bytes={}\n\n",
            t_json.as_secs_f64() * 1000.0,
            (t_prepare - t_json).as_secs_f64() * 1000.0,
            t_entry_to_loop.as_secs_f64() * 1000.0,
            body_len,
        );
        let _ = body.write_all(trace.as_bytes()).await;
        let _ = body.flush().await;
    }

    let mut step_wall_ns: Vec<(u128, u128, u128, f64)> = Vec::new();

    'outer: loop {
        let t0 = std::time::Instant::now();
        let ws = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let dn = if n_yields == 0 { first_decode_n } else { steady_decode_n };
        // Single async yield for up to dn tokens. A per-token decode_step loop
        // here interleaves pathologically with body.flush().await under SSE
        // streaming load (50+ s scheduler stalls observed). See SDK
        // context.rs L985–991 and the decode_stream docstring at L728.
        let tokens = ctx.decode_n(&sampler, dn).await;
        if tokens.is_empty() {
            break 'outer;
        }
        let we = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let step_ms = t0.elapsed().as_secs_f64() * 1000.0;
        t_decode_total += t0.elapsed();
        n_yields += 1;

        let num_tokens = tokens.len();
        for (ti, &token_id) in tokens.iter().enumerate() {
            if ti == num_tokens - 1 {
                ctx.fill_token(token_id);
            }
            generated_token_ids.push(token_id);
            n_steps += 1;

            let t1 = std::time::Instant::now();
            let (delta_text, np, nr) = incremental_detokenize_step(
                |ids| tokenizer.detokenize(ids),
                &generated_token_ids,
                detok_prefix_offset,
                detok_read_offset,
            );
            detok_prefix_offset = np;
            detok_read_offset = nr;
            t_detok_total += t1.elapsed();

            if stop_strings.iter().any(|s| delta_text.contains(s)) {
                // Check stop AFTER filtering so we still break on EOS
                if stop_cond.check(&generated_token_ids) {
                    break 'outer;
                }
                continue;
            }

            full_text.push_str(&delta_text);

            if !in_tool_call && full_text.contains("<tool_call>") {
                in_tool_call = true;
            }

            if !in_tool_call && !delta_text.is_empty() {
                let t2 = std::time::Instant::now();
                let chunk = ChatCompletionChunk {
                    id: chunk_id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created: 0,
                    model: model_name.clone(),
                    pie_cache: None,
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: ChunkDelta {
                            role: None,
                            content: Some(delta_text),
                            tool_calls: None,
                        },
                        finish_reason: None,
                    }],
                };
                let json_str = serde_json::to_string(&chunk).unwrap_or_default();
                t_serial_total += t2.elapsed();

                let line = format!("data: {}\n\n", json_str);

                let t3 = std::time::Instant::now();
                if body.write_all(line.as_bytes()).await.is_err() {
                    return Finished::finish(body, Ok(()), None);
                }
                t_write_total += t3.elapsed();

                let t4 = std::time::Instant::now();
                if body.flush().await.is_err() {
                    return Finished::finish(body, Ok(()), None);
                }
                t_flush_total += t4.elapsed();

                n_emits += 1;
            }

            // Check stop AFTER emit so the last token (e.g., at max_tokens)
            // is included in the streamed output before breaking.
            if stop_cond.check(&generated_token_ids) {
                break 'outer;
            }
        }

        let wf = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        step_wall_ns.push((ws, we, wf, step_ms));

        // NOTE: Do NOT break when tokens.len() < steady_decode_n.
        // Short token batches happen legitimately when:
        //   (a) Python's multi-step loop hits a page boundary (pre-allocated
        //       pages exhausted), or
        //   (b) This request was merged mid-loop into another batch and only
        //       participated in a subset of steps.
        // In both cases generation is NOT complete — stop_cond.check() above
        // already handles EOS and max_tokens.  A short batch simply means
        // "call decode_n again."
    }

    // Emit profiling data as an SSE comment
    let t_loop_total = t_loop_start.elapsed();
    {
        let profile_comment = format!(
            ": [STREAM-PROFILE] steps={} yields={} steps_per_yield={} emits={} loop={:.1}ms decode={:.1}ms detok={:.2}ms serial={:.2}ms write={:.2}ms flush={:.1}ms per_step_decode={:.2}ms per_yield_decode={:.2}ms per_step_other={:.2}ms\n\n",
            n_steps,
            n_yields,
            steady_decode_n,
            n_emits,
            t_loop_total.as_secs_f64() * 1000.0,
            t_decode_total.as_secs_f64() * 1000.0,
            t_detok_total.as_secs_f64() * 1000.0,
            t_serial_total.as_secs_f64() * 1000.0,
            t_write_total.as_secs_f64() * 1000.0,
            t_flush_total.as_secs_f64() * 1000.0,
            t_decode_total.as_secs_f64() * 1000.0 / n_steps.max(1) as f64,
            t_decode_total.as_secs_f64() * 1000.0 / n_yields.max(1) as f64,
            (t_loop_total - t_decode_total).as_secs_f64() * 1000.0 / n_steps.max(1) as f64,
        );
        let _ = body.write_all(profile_comment.as_bytes()).await;
        let _ = body.flush().await;
    }

    // Emit per-step wall-clock timestamps for correlation with engine [TIMING] ws/we
    // Format: step_index ws_ns we_ns decode_ms
    // Can be joined with [TIMING] entries (which also have ws/we) to decompose overhead
    {
        let mut step_lines = String::from(": [STEP-WALL] ");
        for (i, (ws, we, wf, ms)) in step_wall_ns.iter().enumerate() {
            if i > 0 { step_lines.push(' '); }
            step_lines.push_str(&format!("{}:{},{},{}:{:.2}", i, ws, we, wf, ms));
        }
        step_lines.push_str("\n\n");
        let _ = body.write_all(step_lines.as_bytes()).await;
        let _ = body.flush().await;
    }

    // Emit request-level profiling (prepare phase timing)
    {
        let req_profile = format!(
            ": [REQUEST-PROFILE] mode={} format_chat={:.1}ms fill_tokens={:.1}ms flush={:.1}ms import_prefix={:.1}ms validate_prefix={:.1}ms prepare_total={:.1}ms decode_steps={} decode_total={:.1}ms first_step={:.2}ms\n\n",
            prepare_profile.mode,
            prepare_profile.format_chat_ms,
            prepare_profile.fill_tokens_ms,
            prepare_profile.flush_ms,
            prepare_profile.import_prefix_ms,
            prepare_profile.validate_prefix_ms,
            prepare_profile.total_ms,
            n_steps,
            t_decode_total.as_secs_f64() * 1000.0,
            if n_steps > 0 { t_decode_total.as_secs_f64() * 1000.0 / n_steps as f64 } else { 0.0 },
        );
        let _ = body.write_all(req_profile.as_bytes()).await;
        let _ = body.flush().await;
    }

    if let Some(pie_cache) = &pie_cache {
        let cache_comment = format!(
            ": [PIE-CACHE] {}\n\n",
            serde_json::to_string(pie_cache).unwrap_or_default()
        );
        let _ = body.write_all(cache_comment.as_bytes()).await;
        let _ = body.flush().await;
    }

    // Save block-level cache entries from this request's context.
        // Block-cache lookup is disabled (see block_cache.rs top-of-module note),
    // so skip block_cache save too — it would short-circuit session_kv save
    // and break multi-turn KV reuse.  Mirror of the non-streaming path above.
    save_session_kv_state(&ctx, request, session_incoming_tokens.as_deref());

    // [Phase-B instrumentation] Emit the raw generated text as an SSE
    // comment so the observer can diff pie's model output against vLLM's on
    // the same prompt. Escape newlines and carriage returns so SSE framing
    // (`\n\n` separator) stays intact; truncate to 8 KB to bound stream size.
    {
        const RAW_OUTPUT_LIMIT: usize = 8192;
        let mut snippet = full_text.clone();
        if snippet.len() > RAW_OUTPUT_LIMIT {
            snippet.truncate(RAW_OUTPUT_LIMIT);
        }
        let escaped: String = snippet
            .chars()
            .flat_map(|c| match c {
                '\n' => "\\n".chars().collect::<Vec<_>>(),
                '\r' => "\\r".chars().collect::<Vec<_>>(),
                _ => vec![c],
            })
            .collect();
        let comment = format!(
            ": [RAW-OUTPUT] len={} truncated={} text={}\n\n",
            full_text.len(),
            full_text.len() > RAW_OUTPUT_LIMIT,
            escaped,
        );
        let _ = body.write_all(comment.as_bytes()).await;
        let _ = body.flush().await;
    }

    // Parse tool calls from accumulated full text
    let (_, tool_calls) = crate::tool_parser::parse_tool_calls(&full_text);
    let has_tool_calls = !tool_calls.is_empty();

    if has_tool_calls {
        let tc_deltas: Vec<ChunkToolCall> = tool_calls
            .iter()
            .enumerate()
            .map(|(i, tc)| ChunkToolCall {
                index: i as u32,
                id: Some(tc.id.clone()),
                call_type: Some("function".to_string()),
                function: ChunkFunctionCall {
                    name: Some(tc.function.name.clone()),
                    arguments: Some(tc.function.arguments.clone()),
                },
            })
            .collect();

        let tc_chunk = ChatCompletionChunk {
            id: chunk_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: 0,
            model: model_name.clone(),
            pie_cache: None,
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    content: None,
                    tool_calls: Some(tc_deltas),
                },
                finish_reason: None,
            }],
        };
        emit!(serde_json::to_string(&tc_chunk).unwrap_or_default());
    }

    // Final chunk: finish_reason + pie_cache telemetry.
    // pie_cache is attached here (not as a separate chunk) so SSE consumers
    // that serialize chunks into a capture log observe it as a regular data:
    // event. Mirrors the non-streaming response's top-level pie_cache field.
    // The earlier ": [PIE-CACHE] …" comment is retained for scraper-based
    // observers that already rely on it.
    let final_chunk = ChatCompletionChunk {
        id: chunk_id,
        object: "chat.completion.chunk".to_string(),
        created: 0,
        model: model_name,
        pie_cache: pie_cache.clone(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: ChunkDelta {
                role: None,
                content: None,
                tool_calls: None,
            },
            finish_reason: Some(if has_tool_calls {
                "tool_calls".to_string()
            } else {
                "stop".to_string()
            }),
        }],
    };
    emit!(serde_json::to_string(&final_chunk).unwrap_or_default());

    // Done sentinel
    let done_line = "data: [DONE]\n\n";
    let _ = body.write_all(done_line.as_bytes()).await;
    let _ = body.flush().await;

    Finished::finish(body, Ok(()), None)
}

struct PrepareProfile {
    format_chat_ms: f64,
    fill_tokens_ms: f64,
    flush_ms: f64,
    import_prefix_ms: f64,
    validate_prefix_ms: f64,
    total_ms: f64,
    mode: &'static str,
}

impl Default for PrepareProfile {
    fn default() -> Self {
        Self {
            format_chat_ms: 0.0,
            fill_tokens_ms: 0.0,
            flush_ms: 0.0,
            import_prefix_ms: 0.0,
            validate_prefix_ms: 0.0,
            total_ms: 0.0,
            mode: "direct",
        }
    }
}

struct PreparedExecution {
    model: inferlet::Model,
    ctx: inferlet::Context,
    prompt_tokens: u32,
    pie_cache: Option<PieCacheTelemetry>,
    profile: PrepareProfile,
    /// format_chat tokens for this request's input (for session KV export after generation).
    session_incoming_tokens: Option<Vec<u32>>,
}

async fn prepare_execution(
    request: &ChatCompletionRequest,
    variant_header: Option<&str>,
) -> Result<PreparedExecution, String> {
    let t_start = std::time::Instant::now();
    let model = inferlet::get_auto_model();

    // Block-level prefix cache is LAZY: the first session request pays its
    // single full prefill cost, and after generation the post-hook
    // `block_cache::save_ctx_blocks` records per-block chain-hash lookups
    // against that request's OWN exported context.  Any subsequent request
    // whose reordered token sequence shares a leading block prefix hits the
    // cache and imports a truncated slice.

    // Session KV retention path (highest priority)
    if request.pie_session.is_some() {
        return prepare_session_execution(&model, request).await;
    }

    if let Some(pie_prompt) = &request.pie_prompt {
        if pie_prompt.mode == "registered_prefix" {
            return prepare_structured_execution(&model, request, pie_prompt).await;
        }
    }

    // Phase-1 Idea A: variant-KV injection on the non-session, non-registered path.
    // When X-Hermes-Variant is set and valid (not "none"), import the pre-exported
    // variant KV and use it as the prefix instead of a fresh context.
    if let Some(variant) = variant_header {
        if crate::variant::is_valid_variant(variant) && variant != "none" {
            return prepare_variant_execution(&model, request, variant, t_start).await;
        }
    }

    let mut ctx = model.create_context();

    let t0 = std::time::Instant::now();
    let token_ids = crate::prompt_render::format_chat_tokens(
        &model,
        &request.messages,
        &request.tools,
        &request.chat_template_kwargs,
        true,
    )
    .await
    .unwrap_or_default();
    let format_chat_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let prompt_tokens = token_ids.len() as u32;

    let t1 = std::time::Instant::now();
    ctx.fill_tokens(token_ids);
    let fill_tokens_ms = t1.elapsed().as_secs_f64() * 1000.0;

    Ok(PreparedExecution {
        model,
        ctx,
        prompt_tokens,
        pie_cache: Some(PieCacheTelemetry {
            mode: "direct".to_string(),
            cache_epoch: crate::prompt_cache::cache_epoch(),
            fallback_used: false,
            reused_prefix: None,
            stale_prefix: false,
            stale_epoch: false,
            prefill_tokens_skipped: 0,
            prompt_tokens_total: prompt_tokens,
            prompt_tokens_computed: prompt_tokens,
            debug: None,
            session_id: None,
            session_turn: None,
            prefix_match_len: None,
            total_incoming_tokens: None,
            tail_tokens_filled: None,
            fallback_reason: None,
            ephemeral_tokens_appended: None,
        }),
        profile: PrepareProfile {
            format_chat_ms,
            fill_tokens_ms,
            flush_ms: 0.0,
            total_ms: t_start.elapsed().as_secs_f64() * 1000.0,
            mode: "direct",
            ..Default::default()
        },
        session_incoming_tokens: None,
    })
}

/// Phase-1 Idea A: variant-KV injection path.
///
/// Imports pre-exported KV pages for the named variant and reconstructs a
/// Context via `from_imported_state`, then appends the chat tokens on top.
/// This is the non-session, non-registered-prefix path only (Phase-1 scope).
async fn prepare_variant_execution(
    model: &inferlet::Model,
    request: &ChatCompletionRequest,
    variant: &str,
    t_start: std::time::Instant,
) -> Result<PreparedExecution, String> {
    use inferlet::forward::Forward;

    let meta = crate::variant::load_metadata(variant).ok_or_else(|| {
        format!(
            "variant '{}' has no stored metadata — run scripts/export-variant-kv.py first",
            variant
        )
    })?;

    let queue = model.create_queue();
    let handle = crate::variant::handle_name_for(variant);
    let kv_pages = queue.import_kv_pages(&handle);

    if kv_pages.is_empty() {
        return Err(format!(
            "variant '{}' KV pages not found on queue — re-run export-variant-kv.py",
            variant
        ));
    }

    let mut ctx = inferlet::Context::from_imported_state(
        model,
        kv_pages,
        meta.token_ids.clone(),
        meta.kv_page_last_len,
    );

    let t0 = std::time::Instant::now();
    let chat_tokens = crate::prompt_render::format_chat_tokens(
        model,
        &request.messages,
        &request.tools,
        &request.chat_template_kwargs,
        true,
    )
    .await
    .unwrap_or_default();
    let format_chat_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let variant_token_count = meta.token_ids.len() as u32;
    let chat_token_count = chat_tokens.len() as u32;
    let prompt_tokens = variant_token_count + chat_token_count;

    let t1 = std::time::Instant::now();
    ctx.fill_tokens(chat_tokens);
    let fill_tokens_ms = t1.elapsed().as_secs_f64() * 1000.0;

    Ok(PreparedExecution {
        model: model.clone(),
        ctx,
        prompt_tokens,
        pie_cache: Some(PieCacheTelemetry {
            mode: format!("variant:{}", variant),
            cache_epoch: crate::prompt_cache::cache_epoch(),
            fallback_used: false,
            reused_prefix: None,
            stale_prefix: false,
            stale_epoch: false,
            prefill_tokens_skipped: variant_token_count,
            prompt_tokens_total: prompt_tokens,
            prompt_tokens_computed: chat_token_count,
            debug: None,
            session_id: None,
            session_turn: None,
            prefix_match_len: None,
            total_incoming_tokens: None,
            tail_tokens_filled: None,
            fallback_reason: None,
            ephemeral_tokens_appended: None,
        }),
        profile: PrepareProfile {
            format_chat_ms,
            fill_tokens_ms,
            flush_ms: 0.0,
            total_ms: t_start.elapsed().as_secs_f64() * 1000.0,
            mode: "variant",
            ..Default::default()
        },
        session_incoming_tokens: None,
    })
}

async fn prepare_session_execution(
    model: &inferlet::Model,
    request: &ChatCompletionRequest,
) -> Result<PreparedExecution, String> {
    let t_start = std::time::Instant::now();
    let session_id = &request.pie_session.as_ref().unwrap().session_id;

    // Reorder dynamic system sections (## Messaging/## Reactions/## Runtime) to
    // the end before tokenization. This is required for the prefix checkpoint
    // path to work across channels: the checkpoint stores tokens of the
    // reordered stable prefix, and incoming tokens must use the same ordering
    // so `find_prefix_match_len` sees an aligned prefix.  Session KV (Level 1)
    // also uses the reordered form for consistency.
    let reordered_messages = crate::prompt_render::reorder_system_sections(&request.messages);

    // Tokenize WITHOUT generation prompt (one call instead of two).
    // The generation prompt suffix is stable across messages (verified empirically:
    // Qwen2.5 always appends [151644, 77091, 198] = "<|im_start|>assistant\n").
    // We compute the suffix once and append it to derive the full token sequence.
    let t0 = std::time::Instant::now();
    let incoming_tokens_for_match = crate::prompt_render::format_chat_tokens(
        model,
        &reordered_messages,
        &request.tools,
        &request.chat_template_kwargs,
        false,
    )
    .await
    .unwrap_or_default();

    // Get or compute the generation prompt suffix tokens.
    let suffix = match crate::session_cache::load_gen_prompt_suffix() {
        Some(s) => s,
        None => {
            // First request: compute suffix by diffing with/without generation prompt
            let with_gen = crate::prompt_render::format_chat_tokens(
                model,
                &reordered_messages,
                &request.tools,
                &request.chat_template_kwargs,
                true,
            )
            .await
            .unwrap_or_default();
            let suffix = with_gen[incoming_tokens_for_match.len()..].to_vec();
            crate::session_cache::save_gen_prompt_suffix(&suffix);
            suffix
        }
    };

    // Derive the full token sequence by appending the cached suffix.
    let mut incoming_tokens = incoming_tokens_for_match.clone();
    incoming_tokens.extend_from_slice(&suffix);
    let format_chat_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let total_incoming = incoming_tokens.len() as u32;

    // Evict old session if different session_id
    if let Some(active) = crate::session_cache::active_session_id() {
        if active != *session_id {
            crate::session_cache::evict_session(model);
        }
    }

    // --- Level 1: Try session KV ---
    if let Some(session_state) = crate::session_cache::load_session_state(session_id) {
        // Compare against stored tokens (which were tokenized WITHOUT generation prompt).
        // The current incoming_tokens_for_match also excludes generation prompt.
        let match_len = crate::session_cache::find_prefix_match_len(
            &session_state.incoming_tokens,
            &incoming_tokens_for_match,
        );

        if match_len == session_state.incoming_tokens.len() && match_len > 0 {
            let t_import = std::time::Instant::now();
            if let Some(mut ctx) = crate::session_cache::import_session_kv(model, &session_state) {
                let import_ms = t_import.elapsed().as_secs_f64() * 1000.0;
                let tail = incoming_tokens[match_len..].to_vec();
                let tail_len = tail.len() as u32;

                let t_fill = std::time::Instant::now();
                ctx.fill_tokens(tail);
                let fill_ms = t_fill.elapsed().as_secs_f64() * 1000.0;

                return Ok(PreparedExecution {
                    model: model.clone(),
                    ctx,
                    prompt_tokens: total_incoming,
                    pie_cache: Some(PieCacheTelemetry {
                        mode: "session_kv".to_string(),
                        cache_epoch: crate::prompt_cache::cache_epoch(),
                        fallback_used: false,
                        reused_prefix: None,
                        stale_prefix: false,
                        stale_epoch: false,
                        prefill_tokens_skipped: match_len as u32,
                        prompt_tokens_total: total_incoming,
                        prompt_tokens_computed: total_incoming - match_len as u32,
                        debug: None,
                        session_id: Some(session_id.clone()),
                        session_turn: Some(session_state.turn_count + 1),
                        prefix_match_len: Some(match_len as u32),
                        total_incoming_tokens: Some(total_incoming),
                        tail_tokens_filled: Some(tail_len),
                        fallback_reason: None,
                        ephemeral_tokens_appended: None,
                    }),
                    profile: PrepareProfile {
                        format_chat_ms,
                        fill_tokens_ms: fill_ms,
                        import_prefix_ms: import_ms,
                        total_ms: t_start.elapsed().as_secs_f64() * 1000.0,
                        mode: "session_kv",
                        ..Default::default()
                    },
                    session_incoming_tokens: Some(incoming_tokens_for_match.clone()),
                });
            }
        }
    }

    // --- Level 1.5: Try prefix checkpoint (cross-session sharing) ---
    // If ensure_prefix was called (e.g. by the benchmark or bridge at
    // startup), a PrefixCheckpoint is stored.  Import it directly —
    // token-level prefix match is the correctness guard; no hash check
    // needed because there is at most one checkpoint.
    if let Some(checkpoint) = crate::session_cache::load_prefix_checkpoint() {
        let match_len = crate::session_cache::find_prefix_match_len(
            &checkpoint.token_ids,
            &incoming_tokens_for_match,
        );
        if match_len > 0 {
            let t_import = std::time::Instant::now();
            if let Some(mut ctx) = crate::session_cache::import_prefix_checkpoint(model, &checkpoint) {
                let import_ms = t_import.elapsed().as_secs_f64() * 1000.0;
                let tail = incoming_tokens[match_len..].to_vec();
                let tail_len = tail.len() as u32;

                let t_fill = std::time::Instant::now();
                ctx.fill_tokens(tail);
                let fill_ms = t_fill.elapsed().as_secs_f64() * 1000.0;

                return Ok(PreparedExecution {
                    model: model.clone(),
                    ctx,
                    prompt_tokens: total_incoming,
                    pie_cache: Some(PieCacheTelemetry {
                        mode: "prefix_checkpoint".to_string(),
                        cache_epoch: crate::prompt_cache::cache_epoch(),
                        fallback_used: false,
                        reused_prefix: None,
                        stale_prefix: false,
                        stale_epoch: false,
                        prefill_tokens_skipped: match_len as u32,
                        prompt_tokens_total: total_incoming,
                        prompt_tokens_computed: total_incoming - match_len as u32,
                        debug: None,
                        session_id: Some(session_id.clone()),
                        session_turn: Some(1),
                        prefix_match_len: Some(match_len as u32),
                        total_incoming_tokens: Some(total_incoming),
                        tail_tokens_filled: Some(tail_len),
                        fallback_reason: None,
                        ephemeral_tokens_appended: None,
                    }),
                    profile: PrepareProfile {
                        format_chat_ms,
                        fill_tokens_ms: fill_ms,
                        import_prefix_ms: import_ms,
                        total_ms: t_start.elapsed().as_secs_f64() * 1000.0,
                        mode: "prefix_checkpoint",
                        ..Default::default()
                    },
                    session_incoming_tokens: Some(incoming_tokens_for_match.clone()),
                });
            }
        }
    }

    // --- Level 2: Try block-level cache lookup ---
    // Walks the incoming tokens in kv_page_size blocks and finds the
    // longest contiguous prefix that has been cached by any prior
    // request.  Works across channel switches because the chain hash
    // only depends on the CONTENT of the blocks, not on which request
    // originally computed them.
    let page_size = model.get_kv_page_size() as usize;
    if let Some((export_name, pages_hit)) =
        crate::block_cache::lookup_longest_prefix(&incoming_tokens, page_size)
    {
        let matched_tokens = pages_hit * page_size;
        if matched_tokens > 0 && matched_tokens <= incoming_tokens.len() {
            let t_import = std::time::Instant::now();
            if let Some(mut ctx) = crate::block_cache::import_prefix(
                model,
                &export_name,
                pages_hit,
                &incoming_tokens,
                page_size,
            ) {
                let import_ms = t_import.elapsed().as_secs_f64() * 1000.0;
                let tail = incoming_tokens[matched_tokens..].to_vec();
                let tail_len = tail.len() as u32;

                let t_fill = std::time::Instant::now();
                ctx.fill_tokens(tail);
                let fill_ms = t_fill.elapsed().as_secs_f64() * 1000.0;

                return Ok(PreparedExecution {
                    model: model.clone(),
                    ctx,
                    prompt_tokens: total_incoming,
                    pie_cache: Some(PieCacheTelemetry {
                        mode: "session_kv".to_string(),
                        cache_epoch: crate::prompt_cache::cache_epoch(),
                        fallback_used: true,
                        reused_prefix: None,
                        stale_prefix: false,
                        stale_epoch: false,
                        prefill_tokens_skipped: matched_tokens as u32,
                        prompt_tokens_total: total_incoming,
                        prompt_tokens_computed: total_incoming - matched_tokens as u32,
                        debug: None,
                        session_id: Some(session_id.clone()),
                        session_turn: Some(1),
                        prefix_match_len: Some(matched_tokens as u32),
                        total_incoming_tokens: Some(total_incoming),
                        tail_tokens_filled: Some(tail_len),
                        fallback_reason: Some("block_cache_hit".to_string()),
                        ephemeral_tokens_appended: None,
                    }),
                    profile: PrepareProfile {
                        format_chat_ms,
                        fill_tokens_ms: fill_ms,
                        import_prefix_ms: import_ms,
                        total_ms: t_start.elapsed().as_secs_f64() * 1000.0,
                        mode: "block_cache",
                        ..Default::default()
                    },
                    session_incoming_tokens: Some(incoming_tokens_for_match.clone()),
                });
            }
        }
    }

    // --- Level 3: Full prefill ---
    let mut ctx = model.create_context();
    let t_fill = std::time::Instant::now();
    ctx.fill_tokens(incoming_tokens);
    let fill_ms = t_fill.elapsed().as_secs_f64() * 1000.0;

    // NOTE: Checkpoint creation removed from Level 3. Creating a second
    // context (prefix_ctx) while the primary context is alive can cause
    // GPU memory corruption. Checkpoints are created only by the prewarm
    // path (before any request context exists).

    Ok(PreparedExecution {
        model: model.clone(),
        ctx,
        prompt_tokens: total_incoming,
        pie_cache: Some(PieCacheTelemetry {
            mode: "session_kv".to_string(),
            cache_epoch: crate::prompt_cache::cache_epoch(),
            fallback_used: true,
            reused_prefix: None,
            stale_prefix: false,
            stale_epoch: false,
            prefill_tokens_skipped: 0,
            prompt_tokens_total: total_incoming,
            prompt_tokens_computed: total_incoming,
            debug: None,
            session_id: Some(session_id.clone()),
            session_turn: Some(1),
            prefix_match_len: Some(0),
            total_incoming_tokens: Some(total_incoming),
            tail_tokens_filled: Some(total_incoming),
            fallback_reason: Some("full_prefill".to_string()),
            ephemeral_tokens_appended: None,
        }),
        profile: PrepareProfile {
            format_chat_ms,
            fill_tokens_ms: fill_ms,
            total_ms: t_start.elapsed().as_secs_f64() * 1000.0,
            mode: "session_full",
            ..Default::default()
        },
        session_incoming_tokens: Some(incoming_tokens_for_match.clone()),
    })
}

async fn prepare_structured_execution(
    model: &inferlet::Model,
    request: &ChatCompletionRequest,
    pie_prompt: &PiePromptRequest,
) -> Result<PreparedExecution, String> {
    let t_start = std::time::Instant::now();
    let fallback_on_epoch_mismatch = pie_prompt
        .fallback
        .as_ref()
        .and_then(|f| f.on_epoch_mismatch.as_deref())
        .unwrap_or("reject");
    let fallback_on_missing_prefix = pie_prompt
        .fallback
        .as_ref()
        .and_then(|f| f.on_missing_prefix.as_deref())
        .unwrap_or("reject");
    let current_epoch = prompt_cache::cache_epoch();
    let debug_options = pie_prompt.debug.clone();

    if pie_prompt.cache_epoch != current_epoch {
        if fallback_on_epoch_mismatch == "render_full_request" {
            return prepare_legacy_fallback(
                model,
                request,
                Some(PieCacheTelemetry {
                    mode: "registered_prefix".to_string(),
                    cache_epoch: current_epoch,
                    fallback_used: true,
                    reused_prefix: None,
                    stale_prefix: false,
                    stale_epoch: true,
                    prefill_tokens_skipped: 0,
                    // prompt_tokens_total/computed: placeholders, back-filled by prepare_legacy_fallback()
                    prompt_tokens_total: 0,
                    prompt_tokens_computed: 0,
                    debug: None,
                    session_id: None,
                    session_turn: None,
                    prefix_match_len: None,
                    total_incoming_tokens: None,
                    tail_tokens_filled: None,
                    fallback_reason: None,
                    ephemeral_tokens_appended: None,
                }),
            )
            .await;
        }
        return Err("stale cache_epoch for pie_prompt prefix_handle".to_string());
    }

    let prefix = match prompt_cache::get_prefix(&pie_prompt.prefix_handle) {
        Some(prefix) => prefix,
        None => {
            if fallback_on_missing_prefix == "render_full_request" {
                return prepare_legacy_fallback(
                    model,
                    request,
                    Some(PieCacheTelemetry {
                        mode: "registered_prefix".to_string(),
                        cache_epoch: current_epoch,
                        fallback_used: true,
                        reused_prefix: None,
                        stale_prefix: true,
                        stale_epoch: false,
                        prefill_tokens_skipped: 0,
                        prompt_tokens_total: 0,
                        prompt_tokens_computed: 0,
                        debug: None,
                        session_id: None,
                        session_turn: None,
                        prefix_match_len: None,
                        total_incoming_tokens: None,
                        tail_tokens_filled: None,
                        fallback_reason: None,
                        ephemeral_tokens_appended: None,
                    }),
                )
                .await;
            }
            return Err("unknown prefix_handle for pie_prompt".to_string());
        }
    };

    let t_import = std::time::Instant::now();
    let mut ctx = match prompt_cache::import_prefix(model, &pie_prompt.prefix_handle) {
        Some(ctx) => ctx,
        None => {
            if fallback_on_missing_prefix == "render_full_request" {
                return prepare_legacy_fallback(
                    model,
                    request,
                    Some(PieCacheTelemetry {
                        mode: "registered_prefix".to_string(),
                        cache_epoch: current_epoch,
                        fallback_used: true,
                        reused_prefix: None,
                        stale_prefix: true,
                        stale_epoch: false,
                        prefill_tokens_skipped: 0,
                        prompt_tokens_total: 0,
                        prompt_tokens_computed: 0,
                        debug: None,
                        session_id: None,
                        session_turn: None,
                        prefix_match_len: None,
                        total_incoming_tokens: None,
                        tail_tokens_filled: None,
                        fallback_reason: None,
                        ephemeral_tokens_appended: None,
                    }),
                )
                .await;
            }
            return Err(
                "registered prefix exists but its exported KV pages are unavailable".to_string(),
            );
        }
    };
    let import_prefix_ms = t_import.elapsed().as_secs_f64() * 1000.0;

    let t_validate = std::time::Instant::now();
    let expected_prefix_tokens = crate::prompt_render::format_request_prefix_tokens(
        model,
        &request.messages,
        &request.tools,
        &request.chat_template_kwargs,
    )
    .await?;

    if prefix.prefix_tokens != expected_prefix_tokens {
        let debug = if let Some(options) = &debug_options {
            if options.compare_full_state {
                build_prompt_cache_debug_info(
                    model,
                    &ctx,
                    request,
                    &expected_prefix_tokens,
                    options.token_sample.unwrap_or(32),
                )
                .await
                .ok()
            } else {
                None
            }
        } else {
            None
        };

        if fallback_on_missing_prefix == "render_full_request" {
            return prepare_legacy_fallback(
                model,
                request,
                Some(PieCacheTelemetry {
                    mode: "registered_prefix".to_string(),
                    cache_epoch: current_epoch,
                    fallback_used: true,
                    reused_prefix: None,
                    stale_prefix: true,
                    stale_epoch: false,
                    prefill_tokens_skipped: 0,
                    prompt_tokens_total: 0,
                    prompt_tokens_computed: 0,
                    debug,
                    session_id: None,
                    session_turn: None,
                    prefix_match_len: None,
                    total_incoming_tokens: None,
                    tail_tokens_filled: None,
                    fallback_reason: None,
                    ephemeral_tokens_appended: None,
                }),
            )
            .await;
        }
        return Err("registered prefix tokens do not match request prefix render".to_string());
    }

    let tail_messages = if let Some(first_message) = request.messages.first() {
        if first_message.role == "system" {
            if first_message.content != prefix.prefix_text {
                if fallback_on_missing_prefix == "render_full_request" {
                    return prepare_legacy_fallback(
                        model,
                        request,
                        Some(PieCacheTelemetry {
                            mode: "registered_prefix".to_string(),
                            cache_epoch: current_epoch,
                            fallback_used: true,
                            reused_prefix: None,
                            stale_prefix: true,
                            stale_epoch: false,
                            prefill_tokens_skipped: 0,
                            prompt_tokens_total: 0,
                            prompt_tokens_computed: 0,
                            debug: None,
                            session_id: None,
                            session_turn: None,
                            prefix_match_len: None,
                            total_incoming_tokens: None,
                            tail_tokens_filled: None,
                            fallback_reason: None,
                            ephemeral_tokens_appended: None,
                        }),
                    )
                    .await;
                }
                return Err("request system message does not match registered prefix".to_string());
            }
            request.messages[1..].to_vec()
        } else {
            request.messages.clone()
        }
    } else {
        Vec::new()
    };

    let validate_prefix_ms = t_validate.elapsed().as_secs_f64() * 1000.0;

    let t_tail = std::time::Instant::now();
    let tail_tokens = crate::prompt_render::format_chat_tokens(
        model,
        &tail_messages,
        &request.tools,
        &request.chat_template_kwargs,
        true, // add_generation_prompt
    )
    .await?;
    let format_chat_tail_ms = t_tail.elapsed().as_secs_f64() * 1000.0;

    let prompt_tokens = (prefix.prefix_tokens.len() + tail_tokens.len()) as u32;

    let debug = if let Some(options) = &debug_options {
        if options.compare_full_state {
            build_prompt_cache_debug_info(
                model,
                &ctx,
                request,
                &expected_prefix_tokens,
                options.token_sample.unwrap_or(32),
            )
            .await
            .ok()
        } else {
            None
        }
    } else {
        None
    };

    let t_fill = std::time::Instant::now();
    ctx.fill_tokens(tail_tokens);
    let fill_tokens_ms = t_fill.elapsed().as_secs_f64() * 1000.0;

    // No explicit flush — decode_step handles prefill implicitly
    let flush_ms = 0.0;

    Ok(PreparedExecution {
        model: model.clone(),
        ctx,
        prompt_tokens,
        pie_cache: Some(PieCacheTelemetry {
            mode: "registered_prefix".to_string(),
            cache_epoch: current_epoch,
            fallback_used: false,
            reused_prefix: Some(prefix.handle),
            stale_prefix: false,
            stale_epoch: false,
            prefill_tokens_skipped: prefix.prefix_tokens.len() as u32,
            prompt_tokens_total: prompt_tokens,
            prompt_tokens_computed: prompt_tokens - prefix.prefix_tokens.len() as u32,
            debug,
            session_id: None,
            session_turn: None,
            prefix_match_len: None,
            total_incoming_tokens: None,
            tail_tokens_filled: None,
            fallback_reason: None,
            ephemeral_tokens_appended: None,
        }),
        profile: PrepareProfile {
            format_chat_ms: validate_prefix_ms + format_chat_tail_ms,
            fill_tokens_ms,
            flush_ms,
            import_prefix_ms,
            validate_prefix_ms,
            total_ms: t_start.elapsed().as_secs_f64() * 1000.0,
            mode: "structured",
        },
        session_incoming_tokens: None,
    })
}

async fn prepare_legacy_fallback(
    model: &inferlet::Model,
    request: &ChatCompletionRequest,
    pie_cache: Option<PieCacheTelemetry>,
) -> Result<PreparedExecution, String> {
    let mut ctx = model.create_context();
    let prompt_tokens = fill_context(model, &mut ctx, request).await as u32;
    // Back-fill token breakdown now that prompt_tokens is known.
    let pie_cache = pie_cache.map(|mut t| {
        t.prompt_tokens_total = prompt_tokens;
        t.prompt_tokens_computed = prompt_tokens - t.prefill_tokens_skipped;
        t
    });
    Ok(PreparedExecution {
        model: model.clone(),
        ctx,
        prompt_tokens,
        pie_cache,
        profile: PrepareProfile::default(),
        session_incoming_tokens: None,
    })
}

async fn build_prompt_cache_debug_info(
    model: &inferlet::Model,
    imported_prefix_ctx: &inferlet::Context,
    request: &ChatCompletionRequest,
    expected_prefix_tokens: &[u32],
    token_sample: usize,
) -> Result<PromptCacheDebugInfo, String> {
    let mut rendered_prefix_ctx = model.create_context();
    rendered_prefix_ctx.fill_tokens(expected_prefix_tokens.to_vec());
    rendered_prefix_ctx.flush().await;

    let mut structured_full_ctx = imported_prefix_ctx.fork();
    let tail_messages = if let Some(first_message) = request.messages.first() {
        if first_message.role == "system" {
            request.messages[1..].to_vec()
        } else {
            request.messages.clone()
        }
    } else {
        Vec::new()
    };
    let tail_tokens = crate::prompt_render::format_chat_tokens(
        model,
        &tail_messages,
        &request.tools,
        &request.chat_template_kwargs,
        true,
    )
    .await?;
    structured_full_ctx.fill_tokens(tail_tokens);
    structured_full_ctx.flush().await;

    let mut rendered_full_ctx = model.create_context();
    fill_context(model, &mut rendered_full_ctx, request).await;
    rendered_full_ctx.flush().await;

    Ok(PromptCacheDebugInfo {
        prefix_tokens_match: imported_prefix_ctx.get_token_ids()
            == rendered_prefix_ctx.get_token_ids(),
        full_prompt_tokens_match: structured_full_ctx.get_token_ids()
            == rendered_full_ctx.get_token_ids(),
        first_prefix_token_mismatch: first_token_mismatch(
            imported_prefix_ctx.get_token_ids(),
            rendered_prefix_ctx.get_token_ids(),
        ),
        first_full_prompt_token_mismatch: first_token_mismatch(
            structured_full_ctx.get_token_ids(),
            rendered_full_ctx.get_token_ids(),
        ),
        cached_prefix: snapshot_context(imported_prefix_ctx, token_sample),
        rendered_prefix: snapshot_context(&rendered_prefix_ctx, token_sample),
        structured_full: snapshot_context(&structured_full_ctx, token_sample),
        rendered_full: snapshot_context(&rendered_full_ctx, token_sample),
    })
}

fn snapshot_context(ctx: &inferlet::Context, token_sample: usize) -> PromptContextSnapshot {
    PromptContextSnapshot {
        token_count: ctx.get_token_ids().len() as u32,
        kv_page_count: ctx.kv_pages.len() as u32,
        kv_page_last_len: ctx.get_kv_page_last_len() as u32,
        kv_page_ptrs: ctx.get_kv_page_ptrs(),
        mask_current_brle: ctx.token_mask_current.buffer.clone(),
        mask_pending_len: ctx.token_mask_pending.len() as u32,
        position_count: ctx.position_ids.len() as u32,
        token_sample: ctx
            .get_token_ids()
            .iter()
            .take(token_sample)
            .copied()
            .collect(),
    }
}

fn first_token_mismatch(left: &[u32], right: &[u32]) -> Option<u32> {
    let first = left
        .iter()
        .zip(right.iter())
        .position(|(lhs, rhs)| lhs != rhs)
        .map(|idx| idx as u32);
    if first.is_some() {
        return first;
    }
    if left.len() != right.len() {
        return Some(left.len().min(right.len()) as u32);
    }
    None
}

/// Export the current context's KV state for session reuse on the next turn.
/// Export the current context's KV state for session reuse on the next turn.
/// `incoming_tokens` is the format_chat output for THIS turn's input messages —
/// it's what the next turn will compare against (not ctx.get_token_ids() which
/// includes generated tokens that don't round-trip through format_chat).
fn save_session_kv_state(
    ctx: &inferlet::Context,
    request: &ChatCompletionRequest,
    incoming_tokens: Option<&[u32]>,
) {
    let pie_session = match &request.pie_session {
        Some(s) => s,
        None => return,
    };
    let incoming = match incoming_tokens {
        Some(t) => t,
        None => return,
    };

    let session_id = &pie_session.session_id;
    let kv_page_last_len = ctx.get_kv_page_last_len();

    if ctx.kv_pages.is_empty() {
        return;
    }

    // Versioned export names: the runtime keeps exports as shared read-only
    // resources (import doesn't consume), so each turn must use a unique name.
    // Old exports accumulate but are harmless — they reference the same
    // physical pages (a subset of the current turn's pages).  They are
    // cleaned up when the session is evicted (evict_session releases all).
    let turn_count = crate::session_cache::load_session_state(session_id)
        .map(|s| s.turn_count + 1)
        .unwrap_or(1);

    let export_name = format!("session-kv:{}:t{}", session_id, turn_count);

    // Synchronous export: if the engine rejects the ptrs list (e.g. a ref-
    // count accounting edge case removed a ptr from the instance's allocated
    // set between import and export), we observe the Err here and skip
    // save_session_state. The next turn's load_session_state then returns
    // the prior valid state (or None, triggering prefix_checkpoint / full
    // prefill), instead of loading a name the engine has no record of and
    // crashing via the import-of-missing-export path.
    if let Err(e) = ctx.queue.export_kv_pages_sync(&ctx.kv_pages, &export_name) {
        eprintln!("[pieclaw] export_kv_pages_sync failed for {export_name}: {e} — skipping save_session_state");
        return;
    }

    let reordered = crate::prompt_render::reorder_system_sections(&request.messages);
    let system_hash = crate::session_cache::hash_reordered_stable_prefix(&reordered);

    let state = crate::session_cache::SessionKvState {
        session_id: session_id.clone(),
        export_name,
        incoming_tokens: incoming.to_vec(),
        kv_page_last_len,
        turn_count,
        prefix_checkpoint_hash: system_hash,
    };
    crate::session_cache::save_session_state(&state);
}

#[cfg(test)]
mod session_id_tests {
    use super::*;
    use crate::types::{ChatMessage, ChatCompletionRequest};

    fn req_with_system(content: &str) -> ChatCompletionRequest {
        let mut req = ChatCompletionRequest::default();
        req.messages = vec![ChatMessage {
            role: "system".to_string(),
            content: content.to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];
        req
    }

    #[test]
    fn auto_session_id_returns_none_when_bearer_empty() {
        let req = req_with_system("you are a helpful assistant");
        assert_eq!(auto_session_id("", &req), None,
                   "empty bearer must never bucket; otherwise anonymous callers collide");
    }

    #[test]
    fn auto_session_id_returns_none_when_bearer_empty_even_with_empty_system() {
        let mut req = ChatCompletionRequest::default();
        req.messages = vec![];
        assert_eq!(auto_session_id("", &req), None);
    }

    #[test]
    fn auto_session_id_deterministic_with_bearer() {
        let req = req_with_system("you are a helpful assistant");
        let a = auto_session_id("sk-alice", &req).unwrap();
        let b = auto_session_id("sk-alice", &req).unwrap();
        assert_eq!(a, b, "same inputs must produce same id");
        assert!(a.starts_with("auto-"));
        assert_eq!(a.len(), "auto-".len() + 16);
    }

    #[test]
    fn auto_session_id_different_bearers_different_buckets() {
        let req = req_with_system("same system prompt across tenants");
        let alice = auto_session_id("sk-alice", &req).unwrap();
        let bob = auto_session_id("sk-bob", &req).unwrap();
        assert_ne!(alice, bob, "same system prompt must not bucket different bearers together");
    }

    #[test]
    fn auto_session_id_handles_multibyte_at_500_boundary() {
        // Pad ASCII up to byte 499, then insert a 4-byte UTF-8 char that
        // straddles the old 500-byte slice boundary. Must not panic.
        let mut s = "a".repeat(499);
        s.push('𝕏'); // 4-byte codepoint
        s.push_str(&"b".repeat(100));
        let req = req_with_system(&s);
        let id = auto_session_id("sk-bearer", &req).expect("must not panic or return None");
        assert!(id.starts_with("auto-"));
    }

    #[test]
    fn auto_session_id_handles_short_system_prompt() {
        let req = req_with_system("short");
        assert!(auto_session_id("sk-bearer", &req).is_some());
    }

    #[test]
    fn scoped_session_id_returns_none_when_bearer_empty() {
        assert_eq!(scoped_session_id("", "my-session"), None);
    }

    #[test]
    fn scoped_session_id_returns_none_when_raw_empty() {
        assert_eq!(scoped_session_id("sk-alice", ""), None);
    }

    #[test]
    fn scoped_session_id_different_bearers_different_buckets() {
        let alice = scoped_session_id("sk-alice", "conversation-42").unwrap();
        let bob = scoped_session_id("sk-bob", "conversation-42").unwrap();
        assert_ne!(alice, bob,
                   "raw X-Pie-Session-Id alone must NOT grant cross-tenant read access");
        assert!(alice.starts_with("scoped-"));
    }

    #[test]
    fn scoped_and_auto_ids_are_distinguishable() {
        let req = req_with_system("sys");
        let auto = auto_session_id("sk-bearer", &req).unwrap();
        let scoped = scoped_session_id("sk-bearer", "sys").unwrap();
        assert_ne!(auto, scoped, "prefixes keep the two tiers separable in logs");
    }
}

#[cfg(test)]
mod incremental_detokenize_tests {
    use super::*;

    /// Tokens 1, 2, 3 each carry one byte of a 3-byte UTF-8 codepoint
    /// (Chinese "你" = E4 BD A0). Whole detokenization yields a valid string;
    /// any strict prefix yields U+FFFD because the bytes are incomplete.
    /// This mirrors how HuggingFace tokenizers / vLLM detokenizers behave on
    /// partial multi-byte slices.
    fn fake_chinese_split(ids: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            match id {
                1 => bytes.push(0xE4),
                2 => bytes.push(0xBD),
                3 => bytes.push(0xA0),
                4 => bytes.push(b'!'),
                _ => {}
            }
        }
        String::from_utf8(bytes).unwrap_or_else(|err| {
            // Walk valid + replacement, like HF tokenizer's decode(errors="replace").
            String::from_utf8_lossy(err.as_bytes()).into_owned()
        })
    }

    #[test]
    fn defers_partial_codepoint_then_emits_whole_char() {
        let ids = vec![1u32, 2, 3];
        let (d1, p1, r1) = incremental_detokenize_step(fake_chinese_split, &ids[..1], 0, 0);
        assert_eq!(d1, "", "single byte of ‘你’ must NOT emit (would be U+FFFD)");
        assert_eq!((p1, r1), (0, 0), "offsets stay put while we wait");

        let (d2, p2, r2) = incremental_detokenize_step(fake_chinese_split, &ids[..2], p1, r1);
        assert_eq!(d2, "", "two bytes still partial");
        assert_eq!((p2, r2), (0, 0));

        let (d3, p3, r3) = incremental_detokenize_step(fake_chinese_split, &ids[..3], p2, r2);
        assert_eq!(d3, "你", "third byte completes the codepoint");
        assert_eq!((p3, r3), (0, 3));
    }

    #[test]
    fn ascii_emits_per_token() {
        let ids = vec![4u32, 4, 4];
        let mut p = 0;
        let mut r = 0;
        let mut buf = String::new();
        for i in 0..ids.len() {
            let (d, np, nr) = incremental_detokenize_step(fake_chinese_split, &ids[..=i], p, r);
            buf.push_str(&d);
            p = np;
            r = nr;
        }
        assert_eq!(buf, "!!!");
        assert_eq!((p, r), (2, 3));
    }

    #[test]
    fn mixed_ascii_then_chinese() {
        // ! ! 你 (1 ASCII byte each + 3 split-byte tokens for one Chinese char)
        let ids = vec![4u32, 4, 1, 2, 3];
        let mut p = 0;
        let mut r = 0;
        let mut buf = String::new();
        for i in 0..ids.len() {
            let (d, np, nr) = incremental_detokenize_step(fake_chinese_split, &ids[..=i], p, r);
            buf.push_str(&d);
            p = np;
            r = nr;
        }
        assert_eq!(buf, "!!你", "ASCII flushes per token, Chinese flushes only once whole");
    }

    #[test]
    fn empty_token_list_is_noop() {
        let (d, p, r) = incremental_detokenize_step(fake_chinese_split, &[], 0, 0);
        assert_eq!(d, "");
        assert_eq!((p, r), (0, 0));
    }

    #[test]
    fn no_new_tokens_returns_empty() {
        let ids = vec![4u32, 4];
        // Pretend we already emitted both tokens.
        let (d, p, r) = incremental_detokenize_step(fake_chinese_split, &ids, 1, 2);
        assert_eq!(d, "");
        assert_eq!((p, r), (1, 2));
    }
}
