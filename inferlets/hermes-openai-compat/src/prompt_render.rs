use crate::types::{ChatMessage, ChatTemplateKwargs, PromptPrefixSlot, PromptRenderContext, Tool};

/// Section headers that carry per-user / per-channel content and should be
/// moved after the stable persona so prefix KV cache can be reused across
/// different users of the same bearer.
///
/// Current hermes-agent (gateway/session.py::build_session_context_prompt)
/// emits `## Current Session Context` as the dynamic block carrying
/// user_name, chat_id, channel topic, home-channel list, and connected
/// platforms. It is already appended to the tail of the system message via
/// `ephemeral_system_prompt` in run_agent.py:8373-8375, so reorder is a
/// no-op for well-formed hermes-agent traffic — but keep this list in sync
/// with the actual dynamic markers so `hash_reordered_stable_prefix` cuts
/// at the right boundary when callers send content out of order.
///
/// History: older hermes-agent deployments emitted `## Messaging`,
/// `## Reactions`, and `## Runtime` as separate dynamic sections. Those
/// headers no longer appear in current hermes-agent — see
/// `memory/hermes_agent_deployment_shape.md`.
const DYNAMIC_SECTIONS: &[&str] = &["## Current Session Context"];

/// Reorder system message sections so channel-dependent sections appear after
/// stable content. Non-system messages pass through unchanged.
///
/// Sections are delimited by markdown headers (`# ` or `## `). Any section whose
/// header matches one of `DYNAMIC_SECTIONS` is moved to the end, preserving
/// relative order within both the stable and dynamic partitions.
///
/// Note: uses `lines()` + `join("\n")` which normalizes trailing newlines.
/// This is acceptable because only moved sections are affected — the stable
/// prefix (which determines APC cache hits) is unchanged.
pub fn reorder_system_sections(messages: &[ChatMessage]) -> Vec<ChatMessage> {
    messages
        .iter()
        .map(|msg| {
            if msg.role != "system" {
                return msg.clone();
            }

            // Parse into sections. Each section is (header_line_or_empty, full_text_including_header).
            let mut sections: Vec<(Option<&str>, String)> = Vec::new();
            let mut current_header: Option<&str> = None;
            let mut current_lines: Vec<&str> = Vec::new();

            for line in msg.content.lines() {
                let is_header = line.starts_with("## ") || line.starts_with("# ");
                if is_header && !current_lines.is_empty() {
                    // Flush previous section
                    sections.push((current_header, current_lines.join("\n")));
                    current_lines.clear();
                    current_header = Some(line);
                } else if is_header {
                    current_header = Some(line);
                }
                current_lines.push(line);
            }
            // Flush last section
            if !current_lines.is_empty() {
                sections.push((current_header, current_lines.join("\n")));
            }

            // Partition into stable and dynamic
            let mut stable = Vec::new();
            let mut dynamic = Vec::new();
            for (header, text) in sections {
                let is_dynamic = header.map_or(false, |h| {
                    let h_trimmed = h.trim();
                    DYNAMIC_SECTIONS.iter().any(|d| h_trimmed == *d)
                });
                if is_dynamic {
                    dynamic.push(text);
                } else {
                    stable.push(text);
                }
            }

            // If nothing moved, return as-is to avoid unnecessary allocation
            if dynamic.is_empty() {
                return msg.clone();
            }

            let mut reordered = stable;
            reordered.extend(dynamic);
            ChatMessage {
                content: reordered.join("\n"),
                ..msg.clone()
            }
        })
        .collect()
}

/// Format chat messages into token IDs via the engine's server-side HF
/// tokenizer template (`Model::format_chat` → `format_chat_rpc`).
///
/// Rationale: client-side minijinja rendering was brittle — some modern model
/// templates (e.g. Qwen3-30B-A3B-Instruct-2507) rely on Jinja2 features or
/// filters that minijinja v2 doesn't fully replicate, silently producing
/// empty/invalid prompts. Routing through the engine's HF Transformers Jinja2
/// environment eliminates that whole class of compatibility bugs at the cost
/// of one ~5ms RPC round-trip per chat turn (negligible vs TTFT).
///
/// Tools are forwarded to the template so models that support native
/// tool-call formatting (e.g. Qwen3 tools branch) can emit `<tool_call>`
/// blocks the constrained sampler expects.
pub async fn format_chat_tokens(
    model: &inferlet::Model,
    messages: &[ChatMessage],
    tools: &Option<Vec<Tool>>,
    chat_template_kwargs: &Option<ChatTemplateKwargs>,
    add_generation_prompt: bool,
) -> Result<Vec<u32>, String> {
    let messages = reorder_system_sections(messages);
    // Remove OpenClaw's `## Tooling` / `## Tool Call Style` instructions from
    // the system prompt so the Qwen chat template's hermes `<tool_call>`
    // injection is the only tool-call format the model sees. Without this,
    // the model follows OpenClaw's natural-language `exec(command=...)`
    // syntax and emits plain text that `parse_tool_calls()` can't extract,
    // producing `finish_reason=stop` with no structured `tool_calls[]`.
    let messages = strip_tool_sections_from_system(&messages);

    // Wrap with chat_template_kwargs when present so engine-side RPC picks up
    // enable_thinking etc. (it accepts either a bare list or a wrapped object).
    let messages_payload = match chat_template_kwargs {
        Some(kwargs) if kwargs.enable_thinking.is_some() => serde_json::json!({
            "messages": messages,
            "chat_template_kwargs": kwargs,
        }),
        _ => serde_json::to_value(&messages)
            .map_err(|e| format!("messages serialize: {e}"))?,
    };
    let messages_json = serde_json::to_string(&messages_payload)
        .map_err(|e| format!("messages json: {e}"))?;

    let tools_json = match tools {
        Some(ts) => Some(
            serde_json::to_string(ts)
                .map_err(|e| format!("tools json: {e}"))?,
        ),
        None => None,
    };

    let queue = model.create_queue();
    Ok(queue
        .format_chat(&messages_json, tools_json.as_deref(), add_generation_prompt)
        .await)
}

/// Strip `## Tooling` and `## Tool Call Style` sections from the system message.
/// These sections duplicate information that the chat template will inject via
/// the `tools` parameter. Keeps all other system prompt content intact.
///
/// Section boundaries:
/// - A section starts when a line trims to exactly `## Tooling` or
///   `## Tool Call Style` (exact match, not prefix — `## Tooling-foo` is
///   a different section).
/// - A section ends at the next markdown header of any depth (`# `, `## `,
///   `### `, …). This prevents swallowing a following `# Project Context`
///   (a top-level header the reorder step relies on).
fn strip_tool_sections_from_system(messages: &[ChatMessage]) -> Vec<ChatMessage> {
    messages.iter().map(|msg| {
        if msg.role != "system" {
            return msg.clone();
        }
        let mut result_lines: Vec<&str> = Vec::new();
        let mut skipping = false;
        for line in msg.content.lines() {
            let trimmed = line.trim_end();
            if trimmed == "## Tooling" || trimmed == "## Tool Call Style" {
                skipping = true;
                continue;
            }
            // Stop skipping at the next markdown header (any depth).
            if skipping && is_md_header(line) {
                skipping = false;
            }
            if !skipping {
                result_lines.push(line);
            }
        }
        ChatMessage {
            content: result_lines.join("\n"),
            ..msg.clone()
        }
    }).collect()
}

/// Returns true when `line` is a markdown ATX header (`#{1,6} …`).
fn is_md_header(line: &str) -> bool {
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() && bytes[i] == b'#' {
        i += 1;
    }
    i >= 1 && i <= 6 && i < bytes.len() && bytes[i] == b' '
}

/// Format prefix slot messages into token IDs using server-side rendering.
pub async fn format_registered_prefix_tokens(
    model: &inferlet::Model,
    slots: &[PromptPrefixSlot],
    render_context: &Option<PromptRenderContext>,
) -> Result<Vec<u32>, String> {
    let messages = prefix_messages_from_slots(slots)?;
    let render_context = render_context.clone().unwrap_or_default();
    format_chat_tokens(
        model,
        &messages,
        &render_context.tools,
        &render_context.chat_template_kwargs,
        false, // no generation prompt for prefix
    )
    .await
}

/// Format just the system-message prefix of a request into token IDs.
pub async fn format_request_prefix_tokens(
    model: &inferlet::Model,
    messages: &[ChatMessage],
    tools: &Option<Vec<Tool>>,
    chat_template_kwargs: &Option<ChatTemplateKwargs>,
) -> Result<Vec<u32>, String> {
    let prefix_messages = request_prefix_messages(messages);
    format_chat_tokens(
        model,
        &prefix_messages,
        tools,
        chat_template_kwargs,
        false, // no generation prompt for prefix
    )
    .await
}

pub fn request_prefix_messages(messages: &[ChatMessage]) -> Vec<ChatMessage> {
    match messages.first() {
        Some(first) if first.role == "system" => vec![first.clone()],
        _ => Vec::new(),
    }
}

pub fn prefix_messages_from_slots(slots: &[PromptPrefixSlot]) -> Result<Vec<ChatMessage>, String> {
    let mut messages = Vec::with_capacity(slots.len());
    for slot in slots {
        if slot.content.format != "text" {
            return Err(format!(
                "unsupported slot content format for {}: {}",
                slot.placement, slot.content.format
            ));
        }
        let role = role_from_placement(&slot.placement)?;
        messages.push(ChatMessage {
            role,
            content: slot.content.text.clone(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        });
    }
    Ok(messages)
}

fn role_from_placement(placement: &str) -> Result<String, String> {
    let role = placement.split('.').next().unwrap_or_default();
    match role {
        "system" | "user" | "assistant" | "tool" => Ok(role.to_string()),
        _ => Err(format!("unsupported slot placement: {}", placement)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ChatMessage;
    use serde::Serialize;

    /// Template message shape historically used by the minijinja kwargs tests.
    /// Kept test-local now that production rendering is done via engine RPC.
    #[derive(Serialize, Clone, Debug)]
    struct TemplateMessage {
        role: String,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning_content: Option<String>,
    }

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_string(),
            content: content.to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    #[test]
    fn test_reorder_moves_dynamic_sections_after_stable() {
        // Simulate a malformed request where hermes-agent's per-user
        // `## Current Session Context` block landed BEFORE the stable
        // persona. The reorder should float it back to the tail.
        let system = msg(
            "system",
            "## Tooling\ntool stuff\n## Safety\nsafe stuff\n## Current Session Context\nSource: Discord (channel: #general)\n# Project Context\nstable 8K block",
        );
        let user = msg("user", "hello");
        let result = reorder_system_sections(&[system, user]);

        let sys_content = &result[0].content;
        // Stable sections come first
        assert!(sys_content.starts_with("## Tooling\ntool stuff\n## Safety\nsafe stuff"));
        // Project Context comes before the dynamic session-context block
        let ctx_pos = sys_content.find("# Project Context").unwrap();
        let session_pos = sys_content.find("## Current Session Context").unwrap();
        assert!(
            ctx_pos < session_pos,
            "Project Context should come before Current Session Context"
        );
    }

    #[test]
    fn test_reorder_no_dynamic_sections_returns_unchanged() {
        let system = msg("system", "## Tooling\ntool stuff\n# Project Context\nstable");
        let result = reorder_system_sections(&[system.clone()]);
        assert_eq!(result[0].content, system.content);
    }

    #[test]
    fn test_reorder_non_system_messages_pass_through() {
        let user = msg(
            "user",
            "## Current Session Context\nshould not be touched when it's user-role content",
        );
        let result = reorder_system_sections(&[user.clone()]);
        assert_eq!(result[0].content, user.content);
    }

    // --- strip_tool_sections_from_system tests ---
    //
    // Rationale: OpenClaw embeds a `## Tool Call Style` section in its system
    // prompt telling the model to use a custom `exec(command=...)` syntax.
    // When a Qwen chat template separately injects hermes `<tool_call>…</tool_call>`
    // format via the tools[] parameter, the model often follows the natural-language
    // system instruction and emits plain-text `exec(...)`, which pie's parse_tool_calls
    // can't extract. Stripping those sections before rendering forces the hermes
    // format to win.

    #[test]
    fn test_strip_removes_tooling_and_tool_call_style_sections() {
        let system = msg(
            "system",
            "# Project Context\nstable\n## Tooling\nuse exec(command=...)\n## Tool Call Style\nyieldMs=5000 mandatory\n## Safety\nkeep",
        );
        let out = strip_tool_sections_from_system(&[system]);
        let c = &out[0].content;
        assert!(!c.contains("## Tooling"), "Tooling section should be stripped: {c}");
        assert!(!c.contains("## Tool Call Style"), "Tool Call Style section should be stripped: {c}");
        assert!(!c.contains("exec(command"), "Tool-syntax instructions should be stripped: {c}");
        assert!(c.contains("# Project Context"), "Unrelated sections preserved: {c}");
        assert!(c.contains("## Safety"), "## Safety preserved: {c}");
    }

    #[test]
    fn test_strip_leaves_non_system_messages_untouched() {
        let user = msg("user", "## Tooling\nkeep this — it's in user content, not system");
        let out = strip_tool_sections_from_system(&[user.clone()]);
        assert_eq!(out[0].content, user.content);
    }

    #[test]
    fn test_strip_stops_at_next_header() {
        let system = msg(
            "system",
            "## Tool Call Style\nA\nB\nC\n## Next\nkeep me",
        );
        let out = strip_tool_sections_from_system(&[system]);
        let c = &out[0].content;
        assert!(!c.contains("## Tool Call Style"));
        assert!(!c.contains("\nA\n"), "lines inside the stripped section are gone: {c}");
        assert!(c.contains("## Next"));
        assert!(c.contains("keep me"));
    }

    #[test]
    fn test_strip_stops_at_h1_boundary() {
        // Regression: earlier version only broke on "## ", so a following
        // "# Project Context" (h1) was swallowed along with Tooling.
        let system = msg(
            "system",
            "## Tooling\nA\nB\n# Project Context\nstable block\n## Safety\nsafe",
        );
        let out = strip_tool_sections_from_system(&[system]);
        let c = &out[0].content;
        assert!(!c.contains("## Tooling"));
        assert!(!c.contains("\nA\n"));
        assert!(c.contains("# Project Context"), "h1 header must survive: {c}");
        assert!(c.contains("stable block"));
        assert!(c.contains("## Safety"));
    }

    #[test]
    fn test_strip_ignores_inline_tooling_substrings() {
        // "## Tooling-advanced" is a different section; do not strip.
        let system = msg(
            "system",
            "## Tooling-advanced\nkeep me\n## Safety\nsafe",
        );
        let out = strip_tool_sections_from_system(&[system]);
        let c = &out[0].content;
        assert!(c.contains("## Tooling-advanced"), "exact-match only: {c}");
        assert!(c.contains("keep me"));
    }

    #[test]
    fn test_reorder_preserves_content_before_first_header() {
        let system = msg(
            "system",
            "preamble text\n## Current Session Context\nuser=alice\n## Safety\nsafe",
        );
        let result = reorder_system_sections(&[system]);
        let c = &result[0].content;
        // Preamble + Safety should come first, then Current Session Context
        assert!(c.starts_with("preamble text\n## Safety\nsafe"), "got: {}", c);
        assert!(c.ends_with("## Current Session Context\nuser=alice"), "got: {}", c);
    }

    // --- Template rendering tests (kwargs plumbing) ---

    /// A minimal Jinja template that echoes `enable_thinking` so we can
    /// verify the kwarg actually reaches the render context.
    const KWARGS_TEST_TEMPLATE: &str = concat!(
        "{% for msg in messages %}",
        "{{ msg.role }}: {{ msg.content }}\n",
        "{% endfor %}",
        "thinking={{ enable_thinking | default('UNSET') }}",
    );

    fn render_with_kwargs(
        messages: &[TemplateMessage],
        kwargs: &Option<ChatTemplateKwargs>,
        add_gen: bool,
    ) -> Result<String, String> {
        let mut ctx = serde_json::Map::new();
        ctx.insert("messages".into(), serde_json::to_value(messages).unwrap());
        ctx.insert("add_generation_prompt".into(), serde_json::Value::Bool(add_gen));
        ctx.insert("begin_of_sequence".into(), serde_json::Value::Bool(true));
        if let Some(kw) = kwargs {
            if let Some(val) = kw.enable_thinking {
                ctx.insert("enable_thinking".into(), serde_json::Value::Bool(val));
            }
        }
        let mut env = minijinja::Environment::new();
        env.add_template("chat", KWARGS_TEST_TEMPLATE)
            .map_err(|e| format!("{e}"))?;
        let tmpl = env.get_template("chat").map_err(|e| format!("{e}"))?;
        tmpl.render(serde_json::Value::Object(ctx))
            .map_err(|e| format!("{e}"))
    }

    #[test]
    fn test_kwargs_enable_thinking_false_reaches_template() {
        let msgs = vec![TemplateMessage {
            role: "user".into(),
            content: "hello".into(),
            reasoning_content: None,
        }];
        let kwargs = Some(ChatTemplateKwargs { enable_thinking: Some(false) });
        let rendered = render_with_kwargs(&msgs, &kwargs, false).unwrap();
        assert!(
            rendered.contains("thinking=false"),
            "enable_thinking=false must reach the template; got: {rendered}"
        );
    }

    #[test]
    fn test_kwargs_enable_thinking_true_reaches_template() {
        let msgs = vec![TemplateMessage {
            role: "user".into(),
            content: "hi".into(),
            reasoning_content: None,
        }];
        let kwargs = Some(ChatTemplateKwargs { enable_thinking: Some(true) });
        let rendered = render_with_kwargs(&msgs, &kwargs, false).unwrap();
        assert!(rendered.contains("thinking=true"), "got: {rendered}");
    }

    #[test]
    fn test_kwargs_none_uses_template_default() {
        let msgs = vec![TemplateMessage {
            role: "user".into(),
            content: "hi".into(),
            reasoning_content: None,
        }];
        let rendered = render_with_kwargs(&msgs, &None, false).unwrap();
        assert!(
            rendered.contains("thinking=UNSET"),
            "no kwargs → template default 'UNSET'; got: {rendered}"
        );
    }
}
