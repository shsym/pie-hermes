//! Parse <tool_call> tags from model output into structured tool calls.

use crate::types::{FunctionCall, ToolCallMsg};

/// Parse model output text for <tool_call>...</tool_call> blocks.
/// Returns (content_before_tools, parsed_tool_calls).
pub fn parse_tool_calls(text: &str) -> (Option<String>, Vec<ToolCallMsg>) {
    let mut tool_calls = Vec::new();
    let mut content_parts = Vec::new();
    let mut remaining = text;
    let mut call_index = 0u32;

    loop {
        match remaining.find("<tool_call>") {
            Some(start) => {
                let before = remaining[..start].trim();
                if !before.is_empty() {
                    content_parts.push(before.to_string());
                }
                let after_open = &remaining[start + "<tool_call>".len()..];
                match after_open.find("</tool_call>") {
                    Some(end) => {
                        let json_str = after_open[..end].trim();
                        if let Some(tc) = parse_single_tool_call(json_str, call_index) {
                            tool_calls.push(tc);
                            call_index += 1;
                        }
                        remaining = &after_open[end + "</tool_call>".len()..];
                    }
                    None => {
                        content_parts.push(remaining.to_string());
                        break;
                    }
                }
            }
            None => {
                let rest = remaining.trim();
                if !rest.is_empty() {
                    content_parts.push(rest.to_string());
                }
                break;
            }
        }
    }

    let content = if content_parts.is_empty() {
        None
    } else {
        Some(content_parts.join("\n"))
    };

    (content, tool_calls)
}

fn parse_single_tool_call(json_str: &str, _index: u32) -> Option<ToolCallMsg> {
    let value: serde_json::Value = serde_json::from_str(json_str).ok()?;
    let name = value.get("name")?.as_str()?.to_string();
    let arguments = match value.get("arguments") {
        Some(args) if args.is_string() => args.as_str().unwrap().to_string(),
        Some(args) => serde_json::to_string(args).ok()?,
        None => "{}".to_string(),
    };

    Some(ToolCallMsg {
        id: crate::types::generate_id("call_"),
        call_type: "function".to_string(),
        function: FunctionCall { name, arguments },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_tool_calls() {
        let (content, calls) = parse_tool_calls("Hello, how can I help?");
        assert_eq!(content, Some("Hello, how can I help?".to_string()));
        assert!(calls.is_empty());
    }

    #[test]
    fn test_single_tool_call() {
        let text = "I'll check the weather.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"SF\"}}\n</tool_call>";
        let (content, calls) = parse_tool_calls(text);
        assert_eq!(content, Some("I'll check the weather.".to_string()));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert!(calls[0].id.starts_with("call_"));
    }

    #[test]
    fn test_multiple_tool_calls() {
        let text = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"SF\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"get_time\", \"arguments\": {\"tz\": \"PST\"}}\n</tool_call>";
        let (content, calls) = parse_tool_calls(text);
        assert!(content.is_none());
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1].function.name, "get_time");
        assert!(calls[1].id.starts_with("call_"));
        assert_ne!(calls[0].id, calls[1].id);
    }

    #[test]
    fn test_tool_call_only() {
        let text =
            "<tool_call>\n{\"name\": \"search\", \"arguments\": {\"q\": \"rust\"}}\n</tool_call>";
        let (content, calls) = parse_tool_calls(text);
        assert!(content.is_none());
        assert_eq!(calls.len(), 1);
    }
}
