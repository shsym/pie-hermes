//! Tool call format detection, borrowing vLLM's per-model parser registry.
//!
//! vLLM uses 34+ hardcoded parsers selected via `--tool-call-parser`.
//! We replicate this as a static registry keyed by model name patterns,
//! with a special-token-based fallback for unknown models.
//!
//! Format is detected once at first request and cached.
//! The `tools` API parameter drives JSON schema for constrained decoding
//! (per-request), while the format (open/close markers) is model-specific.

/// Detected tool call format for a model.
#[derive(Debug, Clone)]
pub struct ToolCallFormat {
    /// Token ID(s) marking the start of a tool call block.
    pub open_token_ids: Vec<u32>,

    /// Token ID(s) marking the end of a tool call block (empty if implicit).
    pub close_token_ids: Vec<u32>,

    /// Whether open/close are single special tokens (enables `<[ID]>` grammar).
    pub single_special_tokens: bool,

    /// The open marker text (for display/logging).
    pub open_text: String,

    /// The close marker text (empty if no explicit close).
    pub close_text: String,

    /// JSON key for function name (typically "name").
    pub name_key: String,

    /// JSON key for arguments ("arguments" for most, "parameters" for some Llama).
    pub args_key: String,

    /// Which vLLM parser this corresponds to (for logging).
    pub parser_name: String,
}

/// Format definition from the registry (static, no tokenizer needed).
struct FormatDef {
    parser_name: &'static str,
    open_marker: &'static str,
    close_marker: &'static str,  // empty = implicit close
    name_key: &'static str,
    args_key: &'static str,
}

/// vLLM-derived format registry. Maps model name patterns to tool call formats.
///
/// Source: vLLM tool_parsers/ (hermes, llama3_json, mistral, deepseek_v3, etc.)
/// Each entry: (model_name_pattern, format_definition)
const FORMAT_REGISTRY: &[(&str, FormatDef)] = &[
    // Qwen3 (non-coder): <tool_call>/<tool_call> with JSON (hermes-compatible)
    ("qwen3", FormatDef {
        parser_name: "hermes",
        open_marker: "<tool_call>",
        close_marker: "</tool_call>",
        name_key: "name",
        args_key: "arguments",
    }),
    // Qwen2.5: same hermes format
    ("qwen2", FormatDef {
        parser_name: "hermes",
        open_marker: "<tool_call>",
        close_marker: "</tool_call>",
        name_key: "name",
        args_key: "arguments",
    }),
    // Hermes models (NousResearch): same format, different token IDs
    ("hermes", FormatDef {
        parser_name: "hermes",
        open_marker: "<tool_call>",
        close_marker: "</tool_call>",
        name_key: "name",
        args_key: "arguments",
    }),
    // Mistral / Mixtral
    ("mistral", FormatDef {
        parser_name: "mistral",
        open_marker: "[TOOL_CALLS]",
        close_marker: "",
        name_key: "name",
        args_key: "arguments",
    }),
    ("mixtral", FormatDef {
        parser_name: "mistral",
        open_marker: "[TOOL_CALLS]",
        close_marker: "",
        name_key: "name",
        args_key: "arguments",
    }),
    // Llama 3.1 / 3.2 (JSON format)
    ("llama-3", FormatDef {
        parser_name: "llama3_json",
        open_marker: "<|python_tag|>",
        close_marker: "<|eom_id|>",
        name_key: "name",
        args_key: "parameters",
    }),
    // Llama 4 (pythonic — NOT supported for constrained decoding yet)
    // Skipped: requires Python AST parsing, not JSON grammar
    //
    // DeepSeek V3: uses fullwidth Unicode markers
    ("deepseek", FormatDef {
        parser_name: "deepseek_v3",
        open_marker: "<｜tool▁call▁begin｜>",
        close_marker: "<｜tool▁call▁end｜>",
        name_key: "name",
        args_key: "arguments",
    }),
    // Granite (IBM)
    ("granite", FormatDef {
        parser_name: "granite",
        open_marker: "<tool_call>",
        close_marker: "</tool_call>",
        name_key: "name",
        args_key: "arguments",
    }),
];

impl ToolCallFormat {
    /// Detect tool call format for the given model.
    ///
    /// Strategy (in order):
    /// 1. Match model name against vLLM-derived format registry
    /// 2. Fall back to special-token-based detection
    /// 3. Return None if format cannot be determined
    pub async fn detect(model: &inferlet::Model) -> Option<Self> {
        let tokenizer = model.get_tokenizer();
        let model_name = model.get_name().to_lowercase();

        // Step 1: Try registry match
        if let Some(fmt) = Self::from_registry(&model_name, &tokenizer) {
            return Some(fmt);
        }

        // Step 2: Fall back to special-token detection
        Self::from_special_tokens(&tokenizer)
    }

    /// Look up format from the vLLM-derived registry by model name.
    fn from_registry(model_name: &str, tokenizer: &inferlet::Tokenizer) -> Option<Self> {
        let lower = model_name.to_lowercase();

        for (pattern, def) in FORMAT_REGISTRY {
            if lower.contains(pattern) {
                return Self::from_def(def, tokenizer);
            }
        }
        None
    }

    /// Build a ToolCallFormat from a registry definition + tokenizer.
    fn from_def(def: &FormatDef, tokenizer: &inferlet::Tokenizer) -> Option<Self> {
        let open_ids = tokenizer.tokenize(def.open_marker);
        let close_ids = if def.close_marker.is_empty() {
            vec![]
        } else {
            tokenizer.tokenize(def.close_marker)
        };

        // Verify markers tokenize to single special tokens for efficient grammar
        let single = open_ids.len() == 1
            && (close_ids.len() <= 1);

        Some(ToolCallFormat {
            open_token_ids: open_ids,
            close_token_ids: close_ids,
            single_special_tokens: single,
            open_text: def.open_marker.to_string(),
            close_text: def.close_marker.to_string(),
            name_key: def.name_key.to_string(),
            args_key: def.args_key.to_string(),
            parser_name: def.parser_name.to_string(),
        })
    }

    /// Detect format from special tokens when registry has no match.
    fn from_special_tokens(tokenizer: &inferlet::Tokenizer) -> Option<Self> {
        let special_tokens = tokenizer.get_special_tokens();
        let special_names: Vec<String> = special_tokens
            .1
            .iter()
            .filter_map(|b| String::from_utf8(b.clone()).ok())
            .collect();

        // Check known special token patterns (ordered by prevalence)
        let markers: &[(&str, &str, &str, &str, &str)] = &[
            // (open, close, name_key, args_key, parser_name)
            ("<tool_call>", "</tool_call>", "name", "arguments", "hermes"),
            ("[TOOL_CALLS]", "", "name", "arguments", "mistral"),
            ("<|python_tag|>", "<|eom_id|>", "name", "parameters", "llama3_json"),
            ("<|tool_call|>", "", "name", "arguments", "granite"),
        ];

        for &(open, close, nk, ak, pn) in markers {
            if special_names.iter().any(|s| s == open) {
                let open_ids = tokenizer.tokenize(open);
                let has_close = !close.is_empty() && special_names.iter().any(|s| s == close);
                let close_ids = if has_close {
                    tokenizer.tokenize(close)
                } else {
                    vec![]
                };

                let single = open_ids.len() == 1 && close_ids.len() <= 1;
                let close_text = if has_close { close.to_string() } else { String::new() };

                return Some(ToolCallFormat {
                    open_token_ids: open_ids,
                    close_token_ids: close_ids,
                    single_special_tokens: single,
                    open_text: open.to_string(),
                    close_text,
                    name_key: nk.to_string(),
                    args_key: ak.to_string(),
                    parser_name: pn.to_string(),
                });
            }
        }

        None
    }
}
