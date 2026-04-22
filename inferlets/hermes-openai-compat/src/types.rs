//! OpenAI chat completions request/response types.
//!
//! Covers both non-streaming (`ChatCompletionResponse`) and
//! streaming SSE (`ChatCompletionChunk`) formats.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// ID generation
// ---------------------------------------------------------------------------

static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generate a unique ID with the given prefix (e.g. "chatcmpl-").
pub fn generate_id(prefix: &str) -> String {
    let n = ID_COUNTER.fetch_add(1, Ordering::SeqCst);
    format!("{}{:016x}", prefix, n)
}

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

/// Tool choice policy for constrained decoding.
///
/// Mirrors the OpenAI `tool_choice` parameter:
/// - `Auto` (default / omitted / `"auto"`) — grammar allows free text
///   interleaved with tool calls; model decides when to call tools.
/// - `None` (`"none"`) — skip constrained decoding entirely; no tool calls.
/// - `Required` (`"required"`) — grammar forces at least one tool call.
/// - `Function(name)` (`{"type":"function","function":{"name":"…"}}`) —
///   grammar forces exactly one call to the named function.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Function(String),
}

impl Default for ToolChoice {
    fn default() -> Self {
        ToolChoice::Auto
    }
}

/// Deserialize `tool_choice` from the OpenAI polymorphic format.
fn deserialize_tool_choice<'de, D>(deserializer: D) -> Result<ToolChoice, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de;

    struct ToolChoiceVisitor;

    impl<'de> de::Visitor<'de> for ToolChoiceVisitor {
        type Value = ToolChoice;

        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str(r#""none", "auto", "required", or {"type":"function","function":{"name":"…"}}"#)
        }

        fn visit_str<E: de::Error>(self, v: &str) -> Result<ToolChoice, E> {
            match v {
                "none" => Ok(ToolChoice::None),
                "auto" => Ok(ToolChoice::Auto),
                "required" => Ok(ToolChoice::Required),
                other => Err(de::Error::unknown_variant(other, &["none", "auto", "required"])),
            }
        }

        fn visit_map<A: de::MapAccess<'de>>(self, mut map: A) -> Result<ToolChoice, A::Error> {
            // Parse {"type":"function","function":{"name":"…"}}
            let mut fn_name: Option<String> = Option::None;
            while let Some(key) = map.next_key::<String>()? {
                match key.as_str() {
                    "function" => {
                        let inner: serde_json::Value = map.next_value()?;
                        if let Some(name) = inner.get("name").and_then(|n| n.as_str()) {
                            fn_name = Some(name.to_string());
                        }
                    }
                    _ => { let _: serde_json::Value = map.next_value()?; }
                }
            }
            fn_name
                .map(ToolChoice::Function)
                .ok_or_else(|| de::Error::missing_field("function.name"))
        }

        fn visit_none<E: de::Error>(self) -> Result<ToolChoice, E> {
            Ok(ToolChoice::Auto)
        }

        fn visit_unit<E: de::Error>(self) -> Result<ToolChoice, E> {
            Ok(ToolChoice::Auto)
        }
    }

    deserializer.deserialize_any(ToolChoiceVisitor)
}

/// POST /v1/chat/completions request body.
#[derive(Debug, Default, Deserialize)]
pub struct ChatCompletionRequest {
    /// Model name (ignored — we use `get_auto_model()`).
    #[serde(default)]
    #[allow(dead_code)]
    pub model: String,

    /// Conversation messages.
    pub messages: Vec<ChatMessage>,

    /// Whether to stream the response via SSE.
    #[serde(default)]
    pub stream: bool,

    /// Maximum number of tokens to generate.
    #[serde(default)]
    pub max_tokens: Option<usize>,

    /// Sampling temperature.
    #[serde(default)]
    pub temperature: Option<f32>,

    /// Nucleus sampling probability mass.
    #[serde(default)]
    pub top_p: Option<f32>,

    /// Repetition penalty. >1.0 discourages re-emitting tokens from the
    /// recent history. Matches vLLM's semantics; Qwen2.5 ships
    /// `generation_config.json` with `repetition_penalty=1.05`.
    /// Applied by the constrained sampler when tools are present.
    #[serde(default)]
    pub repetition_penalty: Option<f32>,

    /// Top-k sampling cap. 0 or absent disables. Qwen2.5's
    /// `generation_config.json` ships `top_k=20`. Applied by the
    /// constrained sampler when tools are present.
    #[serde(default)]
    pub top_k: Option<u32>,

    /// Tool definitions.
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,

    /// Tool choice policy (`"none"`, `"auto"`, `"required"`, or
    /// `{"type":"function","function":{"name":"…"}}`).
    #[serde(default, deserialize_with = "deserialize_tool_choice")]
    pub tool_choice: ToolChoice,

    /// Optional chat-template-specific kwargs such as Qwen3's enable_thinking.
    #[serde(default)]
    pub chat_template_kwargs: Option<ChatTemplateKwargs>,

    /// Pie-specific prompt-cache extension.
    #[serde(default)]
    pub pie_prompt: Option<PiePromptRequest>,

    /// Pie-specific session KV retention extension.
    #[serde(default)]
    pub pie_session: Option<PieSessionRequest>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ChatTemplateKwargs {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enable_thinking: Option<bool>,
}

/// A single message in the conversation.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    /// Content can be a plain string, an array of content parts, or null.
    /// Arrays are flattened to a single string; null becomes empty string.
    #[serde(deserialize_with = "deserialize_content")]
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallMsg>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Content part in a multi-part message (OpenAI format).
#[derive(Debug, Deserialize)]
struct ContentPart {
    #[serde(default)]
    r#type: String,
    #[serde(default)]
    text: Option<String>,
}

/// Deserializes message content from either a string, an array of content parts, or null.
fn deserialize_content<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de;

    struct ContentVisitor;

    impl<'de> de::Visitor<'de> for ContentVisitor {
        type Value = String;

        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("a string, array of content parts, or null")
        }

        fn visit_str<E: de::Error>(self, v: &str) -> Result<String, E> {
            Ok(v.to_string())
        }

        fn visit_string<E: de::Error>(self, v: String) -> Result<String, E> {
            Ok(v)
        }

        fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<String, A::Error> {
            let mut text = String::new();
            while let Some(part) = seq.next_element::<ContentPart>()? {
                if let Some(t) = part.text {
                    if !text.is_empty() {
                        text.push('\n');
                    }
                    text.push_str(&t);
                }
            }
            Ok(text)
        }

        fn visit_none<E: de::Error>(self) -> Result<String, E> {
            Ok(String::new())
        }

        fn visit_unit<E: de::Error>(self) -> Result<String, E> {
            Ok(String::new())
        }
    }

    deserializer.deserialize_any(ContentVisitor)
}

/// Tool call in an assistant message (OpenAI format).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolCallMsg {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PiePromptRequest {
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub mode: String,
    pub cache_epoch: String,
    pub prefix_handle: String,
    #[serde(default)]
    pub fallback: Option<PiePromptFallback>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub debug: Option<PromptCacheDebugOptions>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct PiePromptFallback {
    #[serde(default)]
    pub on_epoch_mismatch: Option<String>,
    #[serde(default)]
    pub on_missing_prefix: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PieSessionRequest {
    pub session_id: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PromptCacheCapabilitiesResponse {
    pub object: String,
    pub version: String,
    pub features: PromptCacheFeatures,
    pub cache: PromptCacheStateInfo,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PromptCacheFeatures {
    pub registered_prefix_blocks: bool,
    pub conditional_modules: bool,
    pub discontiguous_positions: bool,
    pub scaffolds: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PromptCacheStateInfo {
    pub scope: String,
    pub epoch: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PrefixEnsureRequest {
    pub schema: PromptSchemaRef,
    pub prefix: PromptPrefixDefinition,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub render_context: Option<PromptRenderContext>,
    #[serde(default)]
    pub fallback: Option<serde_json::Value>,
    #[serde(default)]
    pub retention: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct PromptRenderContext {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template_kwargs: Option<ChatTemplateKwargs>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct PromptCacheDebugOptions {
    #[serde(default)]
    pub compare_full_state: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_sample: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PromptSchemaRef {
    pub schema_id: String,
    pub schema_version: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PromptPrefixDefinition {
    #[serde(default)]
    pub prompt_mode: String,
    pub slots: Vec<PromptPrefixSlot>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PromptPrefixSlot {
    pub placement: String,
    pub content: PromptPrefixContent,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PromptPrefixContent {
    pub format: String,
    #[serde(default)]
    pub text: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PrefixEnsureResponse {
    pub object: String,
    pub version: String,
    pub prefix_handle: String,
    pub status: String,
    pub cache: PromptCacheStateInfo,
    pub identity: PromptPrefixIdentity,
    pub slots: Vec<PromptPrefixSlotSummary>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PromptPrefixIdentity {
    pub prefix_hash: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PromptPrefixSlotSummary {
    pub placement: String,
    pub content_hash: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PieCacheTelemetry {
    pub mode: String,
    pub cache_epoch: String,
    pub fallback_used: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reused_prefix: Option<String>,
    pub stale_prefix: bool,
    pub stale_epoch: bool,
    pub prefill_tokens_skipped: u32,
    /// Total tokens in the prompt (after any inferlet transformations like reordering/dedup).
    pub prompt_tokens_total: u32,
    /// Tokens that required GPU prefill computation (= total - skipped).
    pub prompt_tokens_computed: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub debug: Option<PromptCacheDebugInfo>,
    // Session KV fields
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_turn: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_match_len: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_incoming_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tail_tokens_filled: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PromptCacheDebugInfo {
    pub prefix_tokens_match: bool,
    pub full_prompt_tokens_match: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_prefix_token_mismatch: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_full_prompt_token_mismatch: Option<u32>,
    pub cached_prefix: PromptContextSnapshot,
    pub rendered_prefix: PromptContextSnapshot,
    pub structured_full: PromptContextSnapshot,
    pub rendered_full: PromptContextSnapshot,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PromptContextSnapshot {
    pub token_count: u32,
    pub kv_page_count: u32,
    pub kv_page_last_len: u32,
    pub kv_page_ptrs: Vec<u32>,
    pub mask_current_brle: Vec<u32>,
    pub mask_pending_len: u32,
    pub position_count: u32,
    pub token_sample: Vec<u32>,
}

/// Function call details.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

/// Tool definition in the request.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDef,
}

/// Function definition (schema).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionDef {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Non-streaming response types
// ---------------------------------------------------------------------------

/// Full (non-streaming) response for POST /v1/chat/completions.
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pie_cache: Option<PieCacheTelemetry>,
}

/// A single choice in a non-streaming response.
#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: ChoiceMessage,
    pub finish_reason: String,
}

/// The assistant message inside a Choice.
#[derive(Debug, Serialize)]
pub struct ChoiceMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallMsg>>,
}

/// Token usage statistics.
#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ---------------------------------------------------------------------------
// Streaming (SSE) response types
// ---------------------------------------------------------------------------

/// A single SSE chunk for streaming chat completions.
#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

/// A single choice inside a streaming chunk.
#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: ChunkDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// The delta content in a streaming chunk choice.
#[derive(Debug, Serialize)]
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChunkToolCall>>,
}

/// Tool call delta for streaming.
#[derive(Debug, Serialize)]
pub struct ChunkToolCall {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub call_type: Option<String>,
    pub function: ChunkFunctionCall,
}

#[derive(Debug, Serialize)]
pub struct ChunkFunctionCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: deserialize just the tool_choice field from a minimal request JSON.
    fn parse_tool_choice(json: &str) -> ToolChoice {
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        req.tool_choice
    }

    #[test]
    fn tool_choice_default_is_auto() {
        let tc = parse_tool_choice(r#"{"messages":[]}"#);
        assert_eq!(tc, ToolChoice::Auto);
    }

    #[test]
    fn tool_choice_none_string() {
        let tc = parse_tool_choice(r#"{"messages":[],"tool_choice":"none"}"#);
        assert_eq!(tc, ToolChoice::None);
    }

    #[test]
    fn tool_choice_auto_string() {
        let tc = parse_tool_choice(r#"{"messages":[],"tool_choice":"auto"}"#);
        assert_eq!(tc, ToolChoice::Auto);
    }

    #[test]
    fn tool_choice_required_string() {
        let tc = parse_tool_choice(r#"{"messages":[],"tool_choice":"required"}"#);
        assert_eq!(tc, ToolChoice::Required);
    }

    #[test]
    fn tool_choice_function_object() {
        let tc = parse_tool_choice(
            r#"{"messages":[],"tool_choice":{"type":"function","function":{"name":"get_weather"}}}"#,
        );
        assert_eq!(tc, ToolChoice::Function("get_weather".to_string()));
    }

    #[test]
    fn tool_choice_null_is_auto() {
        let tc = parse_tool_choice(r#"{"messages":[],"tool_choice":null}"#);
        assert_eq!(tc, ToolChoice::Auto);
    }
}
