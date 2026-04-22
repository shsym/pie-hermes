//! Convert OpenAI-compatible tool definitions into a Lark grammar for
//! constrained decoding via llguidance.
//!
//! The grammar allows free text interleaved with tool calls.  When the model
//! enters the tool-call open marker (detected from the model's chat template),
//! JSON output is constrained to match one of the declared tool schemas.
//! The close marker (if any) is forced after the JSON body.

use crate::tool_format::ToolCallFormat;
use crate::types::{Tool, ToolChoice};

/// Build a Lark grammar from OpenAI-compatible tool definitions.
///
/// The grammar structure depends on `tool_choice`:
/// - `Auto` — free text interleaved with zero or more tool calls.
/// - `Required` — one or more tool calls (no leading free text).
/// - `Function(name)` — exactly one tool call to the named function.
/// - `None` — caller should skip grammar entirely (not reach here).
///
/// The `format` parameter provides the detected open/close markers and JSON
/// key names, so the grammar works with any model's tool-call convention.
pub fn tools_to_lark_grammar(
    tools: &[Tool],
    format: &ToolCallFormat,
    tool_choice: &ToolChoice,
) -> String {
    let mut grammar = String::new();

    // Top-level rule depends on tool_choice mode
    match tool_choice {
        ToolChoice::Required => {
            grammar.push_str("start: tool_call+\n\n");
        }
        ToolChoice::Function(_) => {
            grammar.push_str("start: tool_call\n\n");
        }
        _ => {
            // Auto (and any unexpected value): free text interleaved with tool calls
            grammar.push_str("start: (free_text | tool_call)*\n\n");
        }
    }

    // Free text rule: only needed for Auto mode (Required/Function don't allow free text)
    if matches!(tool_choice, ToolChoice::Auto) {
        if format.single_special_tokens && format.open_token_ids.len() == 1 {
            grammar.push_str(&format!(
                "free_text: <[^{}]>\n\n",
                format.open_token_ids[0]
            ));
        } else {
            // Multi-token open marker — use lazy regex fallback
            grammar.push_str("free_text[lazy]: /[\\s\\S]+/\n\n");
        }
    }

    // Tool call: open marker, constrained JSON body, close marker (if any).
    // Always use token-ID syntax <[ID]> for special tokens — this avoids
    // ambiguity with Lark syntax (e.g., [TOOL_CALLS] would be parsed as a
    // character class, not a special token reference).
    if format.single_special_tokens && format.open_token_ids.len() == 1 {
        let open_ref = format!("<[{}]>", format.open_token_ids[0]);
        if format.close_token_ids.len() == 1 {
            let close_ref = format!("<[{}]>", format.close_token_ids[0]);
            grammar.push_str(&format!(
                "tool_call: {} tool_json {}\n\n",
                open_ref, close_ref
            ));
        } else {
            // No close token (Mistral-style)
            grammar.push_str(&format!(
                "tool_call: {} tool_json\n\n",
                open_ref
            ));
        }
    } else {
        // Multi-token markers — use quoted literals
        if format.close_text.is_empty() {
            grammar.push_str(&format!(
                "tool_call: \"{}\" tool_json\n\n",
                format.open_text
            ));
        } else {
            grammar.push_str(&format!(
                "tool_call: \"{}\" tool_json \"{}\"\n\n",
                format.open_text, format.close_text
            ));
        }
    }

    // Tool JSON schema — each tool becomes a JSON schema with name/arguments keys
    // determined by the detected format.
    // For Function(name) mode, restrict to just that one tool.
    let effective_tools: Vec<&Tool> = match tool_choice {
        ToolChoice::Function(name) => tools.iter().filter(|t| t.function.name == *name).collect(),
        _ => tools.iter().collect(),
    };
    build_tool_json_rule(&mut grammar, &effective_tools, &format.name_key, &format.args_key);

    grammar
}

/// Build the tool_json grammar rule using the detected JSON key names.
fn build_tool_json_rule(grammar: &mut String, tools: &[&Tool], name_key: &str, args_key: &str) {
    if tools.len() == 1 {
        let schema = build_tool_schema(tools[0], name_key, args_key);
        grammar.push_str(&format!("tool_json: %json {}\n", schema));
    } else {
        // Multiple tools: use JSON Schema anyOf
        let schemas: Vec<String> = tools
            .iter()
            .map(|t| build_tool_schema(t, name_key, args_key))
            .collect();
        let any_of = format!("{{\"anyOf\": [{}]}}", schemas.join(", "));
        grammar.push_str(&format!("tool_json: %json {}\n", any_of));
    }
}

/// Build a JSON schema string for a single tool call object.
///
/// Uses the detected `name_key` and `args_key` (e.g., `"name"` + `"arguments"` for Qwen,
/// `"name"` + `"parameters"` for some Llama models).
fn build_tool_schema(tool: &Tool, name_key: &str, args_key: &str) -> String {
    // Inject additionalProperties:false into the parameter schema to prevent
    // the model from generating extra garbage keys inside tool arguments.
    let params = tool
        .function
        .parameters
        .as_ref()
        .map(|p| {
            let mut schema = p.clone();
            if let Some(obj) = schema.as_object_mut() {
                if obj.get("type").and_then(|t| t.as_str()) == Some("object") {
                    obj.entry("additionalProperties")
                        .or_insert(serde_json::Value::Bool(false));
                }
            }
            schema.to_string()
        })
        .unwrap_or_else(|| r#"{"type":"object","additionalProperties":false}"#.to_string());

    format!(
        r#"{{"type": "object", "properties": {{"{name_key}": {{"const": "{name}"}}, "{args_key}": {params}}}, "required": ["{name_key}", "{args_key}"], "additionalProperties": false}}"#,
        name_key = name_key,
        name = tool.function.name,
        args_key = args_key,
        params = params,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_format::ToolCallFormat;
    use crate::types::{FunctionDef, Tool};

    fn make_tool(name: &str, params: Option<serde_json::Value>) -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: FunctionDef {
                name: name.to_string(),
                description: None,
                parameters: params,
            },
        }
    }

    fn qwen_format() -> ToolCallFormat {
        ToolCallFormat {
            open_token_ids: vec![151657],
            close_token_ids: vec![151658],
            single_special_tokens: true,
            open_text: "<tool_call>".to_string(),
            close_text: "</tool_call>".to_string(),
            name_key: "name".to_string(),
            args_key: "arguments".to_string(),
            parser_name: "hermes".to_string(),
        }
    }

    fn mistral_format() -> ToolCallFormat {
        ToolCallFormat {
            open_token_ids: vec![5],
            close_token_ids: vec![],
            single_special_tokens: true,
            open_text: "[TOOL_CALLS]".to_string(),
            close_text: String::new(),
            name_key: "name".to_string(),
            args_key: "arguments".to_string(),
            parser_name: "mistral".to_string(),
        }
    }

    #[test]
    fn single_tool_grammar_contains_tool_name() {
        let tools = vec![make_tool("search", Some(serde_json::json!({
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        })))];
        let grammar = tools_to_lark_grammar(&tools, &qwen_format(), &ToolChoice::Auto);
        assert!(grammar.contains("search"));
        assert!(grammar.contains("<[151657]>"));  // open token by ID
        assert!(grammar.contains("<[151658]>"));  // close token by ID
        assert!(grammar.contains("%json"));
        assert!(!grammar.contains("anyOf"));
    }

    #[test]
    fn multiple_tools_uses_any_of() {
        let tools = vec![
            make_tool("search", None),
            make_tool("calculate", None),
        ];
        let grammar = tools_to_lark_grammar(&tools, &qwen_format(), &ToolChoice::Auto);
        assert!(grammar.contains("anyOf"));
        assert!(grammar.contains("search"));
        assert!(grammar.contains("calculate"));
    }

    #[test]
    fn tool_without_params_uses_empty_object() {
        let tools = vec![make_tool("ping", None)];
        let format = qwen_format();
        let grammar = tools_to_lark_grammar(&tools, &format, &ToolChoice::Auto);
        let schema = build_tool_schema(&tools[0], &format.name_key, &format.args_key);
        assert!(schema.contains(r#""arguments":"#));
        assert!(grammar.contains("ping"));
    }

    #[test]
    fn mistral_format_no_close_tag() {
        let tools = vec![make_tool("search", None)];
        let grammar = tools_to_lark_grammar(&tools, &mistral_format(), &ToolChoice::Auto);
        assert!(grammar.contains("<[5]>"));  // Mistral open token by ID
        assert!(!grammar.contains("<[151658]>"));  // No Qwen close token
        // Should have tool_call rule without close marker
        assert!(grammar.contains("tool_call: <[5]> tool_json\n"));
    }

    #[test]
    fn custom_json_keys() {
        let tools = vec![make_tool("search", None)];
        let format = ToolCallFormat {
            open_token_ids: vec![100],
            close_token_ids: vec![101],
            single_special_tokens: true,
            open_text: "<|python_tag|>".to_string(),
            close_text: "<|eom_id|>".to_string(),
            name_key: "name".to_string(),
            args_key: "parameters".to_string(),
            parser_name: "llama3_json".to_string(),
        };
        let grammar = tools_to_lark_grammar(&tools, &format, &ToolChoice::Auto);
        let schema = build_tool_schema(&tools[0], &format.name_key, &format.args_key);
        assert!(schema.contains(r#""parameters":"#));
        assert!(!schema.contains(r#""arguments""#));
        assert!(grammar.contains("<[100]>"));  // open token by ID
        assert!(grammar.contains("<[101]>"));  // close token by ID
    }

    #[test]
    fn required_mode_forces_tool_call_plus() {
        let tools = vec![make_tool("search", None)];
        let grammar = tools_to_lark_grammar(&tools, &qwen_format(), &ToolChoice::Required);
        assert!(grammar.contains("start: tool_call+"));
        assert!(!grammar.contains("free_text"));
    }

    #[test]
    fn function_mode_forces_single_tool_call() {
        let tools = vec![
            make_tool("search", None),
            make_tool("calculate", None),
        ];
        let grammar = tools_to_lark_grammar(
            &tools,
            &qwen_format(),
            &ToolChoice::Function("search".to_string()),
        );
        assert!(grammar.contains("start: tool_call\n"));
        assert!(!grammar.contains("free_text"));
        assert!(grammar.contains("search"));
        assert!(!grammar.contains("calculate"));
        assert!(!grammar.contains("anyOf"));
    }

    #[test]
    fn function_mode_unknown_name_produces_empty_schema() {
        let tools = vec![make_tool("search", None)];
        let grammar = tools_to_lark_grammar(
            &tools,
            &qwen_format(),
            &ToolChoice::Function("nonexistent".to_string()),
        );
        // No tools match → tool_json rule should use anyOf with empty list
        assert!(grammar.contains("start: tool_call\n"));
        assert!(!grammar.contains("search"));
    }

    #[test]
    fn auto_mode_allows_free_text() {
        let tools = vec![make_tool("search", None)];
        let grammar = tools_to_lark_grammar(&tools, &qwen_format(), &ToolChoice::Auto);
        assert!(grammar.contains("start: (free_text | tool_call)*"));
        assert!(grammar.contains("free_text"));
    }
}
