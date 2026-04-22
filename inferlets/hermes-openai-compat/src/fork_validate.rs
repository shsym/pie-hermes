//! Fork-based output validation using Pie's zero-copy context forking.
//!
//! Generates N candidates from forked contexts (sharing the same prefix KV cache),
//! then selects the best one based on tool-call validity and token efficiency.
//! This exploits Pie's `ctx.fork()` which is zero-copy — N candidates share the
//! prefix KV, using ~N× less memory than vLLM's `best_of` which duplicates KV.

use crate::stop_conditions::{DegenerateLoopDetector, DynStopCondition, ToolCallBudget, ToolCallComplete};
use inferlet::stop_condition::{StopCondition, ends_with_any, max_len};
use inferlet::{Context, Model, Sampler};

/// Result of fork-based multi-candidate generation.
pub struct ValidatedResult {
    /// The selected candidate text.
    pub text: String,
    /// Index of the chosen candidate (0-based).
    pub candidate_index: usize,
    /// Total number of candidates generated.
    pub n_candidates: usize,
    /// Total tokens generated across all candidates (sum of all candidate lengths).
    pub total_generated_tokens: usize,
}

/// Generate N candidates via zero-copy forking, select the best one.
///
/// Each candidate gets a slightly varied temperature to promote diversity.
/// Candidates are generated sequentially (each fork runs generate() which is async).
///
/// Requirements:
/// - `ctx` must have been flushed (ctx.flush().await) before calling this.
/// - `max_tokens`, `temperature`, `top_p` are the base generation parameters.
/// - `adaptive_stop` controls whether agentic stop conditions are added.
pub async fn generate_best_of_n(
    ctx: &Context,
    model: &Model,
    n: usize,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    adaptive_stop: bool,
) -> ValidatedResult {
    let n = n.clamp(1, 8); // Safety cap

    let mut candidates: Vec<String> = Vec::with_capacity(n);
    let stop_strings = eos_token_strings(model);

    for i in 0..n {
        // Vary temperature across candidates for diversity.
        // Candidate 0 uses base temperature, others get +0.15 increments.
        let t = (temperature + i as f32 * 0.15).min(2.0);

        let mut forked = ctx.fork();
        let sampler = Sampler::top_p(t, top_p);
        let stop_cond = build_candidate_stop_condition(max_tokens, adaptive_stop, model);

        let generated = forked.generate(sampler, stop_cond).await;
        let generated = strip_stop_tokens(&generated, &stop_strings);
        candidates.push(generated);
    }

    let best_idx = select_best_candidate(&candidates);
    let tokenizer = model.get_tokenizer();
    let total_generated_tokens: usize = candidates.iter()
        .map(|c| tokenizer.tokenize(c).len())
        .sum();

    ValidatedResult {
        text: candidates.swap_remove(best_idx),
        candidate_index: best_idx,
        n_candidates: n,
        total_generated_tokens,
    }
}

/// Build a stop condition for a single candidate (mirrors handler's build_stop_condition).
fn build_candidate_stop_condition(
    max_tokens: usize,
    adaptive_stop: bool,
    model: &Model,
) -> DynStopCondition {
    let base = max_len(max_tokens).or(ends_with_any(model.eos_tokens()));
    let mut cond = DynStopCondition::new(base);
    if adaptive_stop {
        let tokenizer = model.get_tokenizer();
        cond = cond
            .or_dyn(ToolCallBudget::new(&tokenizer, 3))
            .or_dyn(ToolCallComplete::new(&tokenizer))
            .or_dyn(DegenerateLoopDetector::default());
    }
    cond
}

/// Select the best candidate index.
///
/// Strategy:
/// 1. Prefer candidates with valid tool calls (parseable JSON inside `<tool_call>` tags).
/// 2. Among valid candidates, pick the shortest (most token-efficient).
/// 3. If no candidates have tool calls at all, pick the shortest.
/// 4. If all tool calls are invalid, still pick the shortest (best we can do).
fn select_best_candidate(candidates: &[String]) -> usize {
    if candidates.len() == 1 {
        return 0;
    }

    let validities: Vec<ToolCallValidity> = candidates.iter().map(|c| check_tool_calls(c)).collect();

    // If any candidate has valid tool calls, filter to those.
    let has_any_tool_calls = validities.iter().any(|v| matches!(v, ToolCallValidity::Valid | ToolCallValidity::Invalid));

    if has_any_tool_calls {
        // Prefer valid tool call candidates; among those, pick shortest.
        let valid_indices: Vec<usize> = validities
            .iter()
            .enumerate()
            .filter(|(_, v)| matches!(v, ToolCallValidity::Valid))
            .map(|(i, _)| i)
            .collect();

        if !valid_indices.is_empty() {
            return *valid_indices
                .iter()
                .min_by_key(|&&i| candidates[i].len())
                .unwrap();
        }
    }

    // No valid tool calls (or no tool calls at all): pick shortest candidate.
    candidates
        .iter()
        .enumerate()
        .min_by_key(|(_, c)| c.len())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[derive(Debug)]
enum ToolCallValidity {
    /// No `<tool_call>` tags present — free text response, considered acceptable.
    NoToolCalls,
    /// Has `<tool_call>...</tool_call>` with parseable JSON inside.
    Valid,
    /// Has `<tool_call>` tags but JSON is malformed.
    Invalid,
}

/// Check whether a candidate's tool calls (if any) contain valid JSON.
fn check_tool_calls(text: &str) -> ToolCallValidity {
    let open_tag = "<tool_call>";
    let close_tag = "</tool_call>";

    let mut found_any = false;
    let mut search_from = 0;

    while let Some(start) = text[search_from..].find(open_tag) {
        let abs_start = search_from + start + open_tag.len();
        if let Some(end) = text[abs_start..].find(close_tag) {
            found_any = true;
            let json_str = text[abs_start..abs_start + end].trim();
            if serde_json::from_str::<serde_json::Value>(json_str).is_err() {
                return ToolCallValidity::Invalid;
            }
            search_from = abs_start + end + close_tag.len();
        } else {
            // Unclosed tool_call tag — invalid.
            return ToolCallValidity::Invalid;
        }
    }

    if found_any {
        ToolCallValidity::Valid
    } else {
        ToolCallValidity::NoToolCalls
    }
}

// Reuse shared helpers from handler to avoid duplication.
use crate::handler::{eos_token_strings, strip_stop_tokens};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_tool_calls_no_tags() {
        assert!(matches!(
            check_tool_calls("Hello, how can I help?"),
            ToolCallValidity::NoToolCalls
        ));
    }

    #[test]
    fn test_check_tool_calls_valid() {
        let text = r#"<tool_call>{"name":"search","arguments":{"q":"test"}}</tool_call>"#;
        assert!(matches!(check_tool_calls(text), ToolCallValidity::Valid));
    }

    #[test]
    fn test_check_tool_calls_invalid_json() {
        let text = "<tool_call>{broken json</tool_call>";
        assert!(matches!(check_tool_calls(text), ToolCallValidity::Invalid));
    }

    #[test]
    fn test_check_tool_calls_unclosed() {
        let text = "<tool_call>{\"name\":\"test\"}";
        assert!(matches!(check_tool_calls(text), ToolCallValidity::Invalid));
    }

    #[test]
    fn test_select_best_single() {
        assert_eq!(select_best_candidate(&["hello".to_string()]), 0);
    }

    #[test]
    fn test_select_best_prefers_valid_tool_call() {
        let candidates = vec![
            "Some long rambling text without tool calls that goes on and on".to_string(),
            r#"<tool_call>{"name":"search","arguments":{}}</tool_call>"#.to_string(),
            "<tool_call>{broken</tool_call>".to_string(),
        ];
        assert_eq!(select_best_candidate(&candidates), 1);
    }

    #[test]
    fn test_select_best_shortest_when_no_tool_calls() {
        let candidates = vec![
            "This is a longer response with more text".to_string(),
            "Short answer".to_string(),
            "Medium length response".to_string(),
        ];
        assert_eq!(select_best_candidate(&candidates), 1);
    }

    #[test]
    fn test_select_best_shortest_among_valid() {
        let candidates = vec![
            r#"<tool_call>{"name":"a","arguments":{"very":"long","list":"of","params":"here"}}</tool_call>"#.to_string(),
            r#"<tool_call>{"name":"a","arguments":{}}</tool_call>"#.to_string(),
        ];
        assert_eq!(select_best_candidate(&candidates), 1);
    }
}
