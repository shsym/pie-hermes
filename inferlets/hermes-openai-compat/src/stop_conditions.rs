//! Composable stop conditions for agentic workloads.
//!
//! These extend the base `StopCondition` trait from the inferlet SDK with
//! domain-specific conditions for tool-calling LLM patterns:
//!
//! - **ToolCallComplete** — stops trailing text after the last `</tool_call>` close tag
//! - **DegenerateLoopDetector** — stops when an N-gram repeats K times
//! - **ToolCallBudget** — stops after N complete tool calls detected

use inferlet::stop_condition::StopCondition;

// ---------------------------------------------------------------------------
// ToolCallComplete
// ---------------------------------------------------------------------------

/// Stops generation when the model produces trailing text after completing tool calls.
///
/// Detects the pattern where a model finishes one or more `<tool_call>...</tool_call>`
/// blocks and then starts generating non-tool-call content (rambling, explanations,
/// etc.). Fires when more than `trailing_threshold` tokens appear after the last
/// `</tool_call>` without a new `<tool_call>` opening.
///
/// This does NOT stop after the first `</tool_call>` — that's handled by `ToolCallBudget`.
pub struct ToolCallComplete {
    open_tokens: Vec<u32>,
    close_tokens: Vec<u32>,
    /// How many non-tag tokens after last </tool_call> before we trigger stop.
    trailing_threshold: usize,
}

impl ToolCallComplete {
    pub fn new(tokenizer: &inferlet::Tokenizer) -> Self {
        Self {
            open_tokens: tokenizer.tokenize("<tool_call>"),
            close_tokens: tokenizer.tokenize("</tool_call>"),
            trailing_threshold: 20,
        }
    }
}

impl StopCondition for ToolCallComplete {
    fn check(&self, token_ids: &[u32]) -> bool {
        // Find the position after the last </tool_call>
        let close_len = self.close_tokens.len();
        if close_len == 0 || token_ids.len() < close_len {
            return false;
        }

        // Scan for the last occurrence of </tool_call>
        let mut last_close_end: Option<usize> = None;
        for i in (0..=(token_ids.len() - close_len)).rev() {
            if token_ids[i..i + close_len] == self.close_tokens[..] {
                last_close_end = Some(i + close_len);
                break;
            }
        }

        let last_close_end = match last_close_end {
            Some(pos) => pos,
            None => return false, // No complete tool call yet
        };

        // Check tokens after the last </tool_call>
        let trailing = &token_ids[last_close_end..];
        if trailing.len() <= self.trailing_threshold {
            return false; // Not enough trailing tokens yet
        }

        // If a new <tool_call> opens after the last close, the model is making
        // another tool call — don't stop.
        let open_len = self.open_tokens.len();
        if open_len > 0 && trailing.len() >= open_len {
            if contains_subsequence(trailing, &self.open_tokens) {
                return false;
            }
        }

        // More than threshold tokens after last </tool_call> with no new <tool_call>:
        // the model is generating trailing garbage — stop.
        true
    }
}

// ---------------------------------------------------------------------------
// DegenerateLoopDetector
// ---------------------------------------------------------------------------

/// Stops generation when the same N-gram repeats K times consecutively.
///
/// This catches degenerate repetition loops that waste tokens without producing
/// useful output (common in tool-calling models when they get stuck).
pub struct DegenerateLoopDetector {
    ngram_size: usize,
    repeat_threshold: usize,
}

impl DegenerateLoopDetector {
    pub fn new(ngram_size: usize, repeat_threshold: usize) -> Self {
        Self {
            ngram_size,
            repeat_threshold,
        }
    }
}

impl Default for DegenerateLoopDetector {
    fn default() -> Self {
        Self::new(10, 3)
    }
}

impl StopCondition for DegenerateLoopDetector {
    fn check(&self, token_ids: &[u32]) -> bool {
        let window = self.ngram_size * self.repeat_threshold;
        if token_ids.len() < window {
            return false;
        }
        let tail = &token_ids[token_ids.len() - window..];
        let ngram = &tail[..self.ngram_size];
        // Check that every subsequent chunk matches the first ngram
        for i in 1..self.repeat_threshold {
            let chunk = &tail[i * self.ngram_size..(i + 1) * self.ngram_size];
            if chunk != ngram {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// ToolCallBudget
// ---------------------------------------------------------------------------

/// Stops generation after N complete tool calls are detected.
///
/// Counts occurrences of the `</tool_call>` close tag in the token stream.
/// Once the count reaches `max_calls`, generation stops to prevent runaway
/// tool-calling loops.
pub struct ToolCallBudget {
    close_tokens: Vec<u32>,
    max_calls: usize,
}

impl ToolCallBudget {
    pub fn new(tokenizer: &inferlet::Tokenizer, max_calls: usize) -> Self {
        Self {
            close_tokens: tokenizer.tokenize("</tool_call>"),
            max_calls,
        }
    }
}

impl StopCondition for ToolCallBudget {
    fn check(&self, token_ids: &[u32]) -> bool {
        if self.close_tokens.is_empty() || token_ids.len() < self.close_tokens.len() {
            return false;
        }
        let mut count = 0usize;
        let len = self.close_tokens.len();
        // Slide through token_ids looking for close tag matches
        for i in 0..=(token_ids.len() - len) {
            if token_ids[i..i + len] == self.close_tokens[..] {
                count += 1;
                if count >= self.max_calls {
                    return true;
                }
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// Boxed wrapper for dynamic dispatch
// ---------------------------------------------------------------------------

/// Wrapper enabling dynamic dispatch for stop conditions.
///
/// The SDK's `generate()` method is generic over `S: StopCondition`, so we need
/// a concrete type that can hold any combination of conditions. This wraps a
/// `Box<dyn StopCondition>` and implements `StopCondition` for it.
pub struct DynStopCondition {
    inner: Box<dyn StopCondition>,
}

impl DynStopCondition {
    pub fn new<S: StopCondition + 'static>(cond: S) -> Self {
        Self {
            inner: Box::new(cond),
        }
    }

    /// Combine with another condition (logical OR), returning a new DynStopCondition.
    pub fn or_dyn<S: StopCondition + 'static>(self, other: S) -> Self {
        Self::new(DynOr {
            first: self,
            second: DynStopCondition::new(other),
        })
    }
}

impl StopCondition for DynStopCondition {
    fn check(&self, token_ids: &[u32]) -> bool {
        self.inner.check(token_ids)
    }
}

/// Internal combinator for DynStopCondition.
struct DynOr {
    first: DynStopCondition,
    second: DynStopCondition,
}

impl StopCondition for DynOr {
    fn check(&self, token_ids: &[u32]) -> bool {
        self.first.check(token_ids) || self.second.check(token_ids)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Check if `haystack` contains `needle` as a contiguous subsequence.
fn contains_subsequence(haystack: &[u32], needle: &[u32]) -> bool {
    if needle.is_empty() {
        return true;
    }
    if haystack.len() < needle.len() {
        return false;
    }
    haystack.windows(needle.len()).any(|w| w == needle)
}
