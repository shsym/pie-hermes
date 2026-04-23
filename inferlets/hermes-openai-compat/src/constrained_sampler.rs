//! A custom sampler that constrains token generation to match a Lark grammar.
//!
//! Uses the `llguidance` library to compute token masks based on grammar state,
//! ensuring outputs are always syntactically valid.
//!
//! Ported from pie/sdk/examples/constrained-decoding/src/sampler.rs.

use fancy_regex::Regex;
use inferlet::sampler::{
    EmittedHistory, Sample, apply_repetition_penalty, apply_top_k, apply_top_p, weighted_sample,
};
use inferlet::{Result, bail};
use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::{TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv};
use llguidance::{Matcher, ParserFactory};
use rand_core::RngCore;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

pub type Rank = u32;

pub struct ConstrainedSampler {
    inner: RefCell<Inner>,
}

struct Inner {
    constraint: Matcher,
    eos_token_id: u32,
    /// xorshift64* PRNG wrapped in a `RngCore` adapter so it can feed the
    /// SDK's `weighted_sample`. Seeded from WASI random (or host time
    /// outside WASM) at construction.
    rng: XorshiftRng,
    /// Multiplicative divisor applied to the probability of any token
    /// present in `history` before the weighted draw. 1.0 disables the
    /// penalty (identity). Values >1 make repeats less likely.
    ///
    /// Matches vLLM's `repetition_penalty` semantics; Qwen2.5's
    /// `generation_config.json` ships `repetition_penalty=1.05`.
    rep_penalty: f32,
    /// Nucleus (top-p) cutoff, in (0.0, 1.0]. Values >= 1.0 disable.
    top_p: f32,
    /// Top-k cap. 0 disables.
    top_k: u32,
    /// Bounded sliding-window tracker (SDK primitive) for recently emitted
    /// tokens. Backed by a `VecDeque` + ref-count `HashMap` for O(1)
    /// `contains` — fed to `apply_repetition_penalty` each step.
    history: EmittedHistory,
}

type Vocab = (Vec<u32>, Vec<Vec<u8>>);

/// Default history window for repetition tracking. 256 tokens covers
/// typical tool-call alternation distances (each `<tool_call>{…}</tool_call>`
/// block is ~30-60 tokens) without blowing up the linear-scan cost.
const DEFAULT_HISTORY_MAX: usize = 256;

/// Default seed hint used when entropy is unavailable (tests on non-wasi
/// hosts, or a zero u64 bubbling up). Golden ratio constant.
const XORSHIFT_SEED_FALLBACK: u64 = 0x9E3779B97F4A7C15;

#[cfg(target_arch = "wasm32")]
fn initial_seed() -> u64 {
    // WASI cryptographic RNG. Always returns fresh entropy.
    let s = wasip2::random::random::get_random_u64();
    if s == 0 { XORSHIFT_SEED_FALLBACK } else { s }
}

#[cfg(not(target_arch = "wasm32"))]
fn initial_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| {
            // Mix seconds and nanos so tests run back-to-back still differ.
            (d.as_secs().wrapping_mul(0x9E37_79B9) ^ d.subsec_nanos() as u64)
                .max(1)
        })
        .unwrap_or(XORSHIFT_SEED_FALLBACK)
}

/// xorshift64*. Fast, good-enough non-crypto PRNG. Returns a u64 and
/// advances the state in place.
#[inline]
fn xorshift64_next(state: &mut u64) -> u64 {
    let mut x = *state;
    if x == 0 {
        x = XORSHIFT_SEED_FALLBACK;
    }
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

/// `rand_core::RngCore` adapter over the xorshift64* state. We keep our own
/// PRNG (rather than pulling in `rand_chacha`) because it's lighter in WASM
/// and already entropy-seeded via WASI. The adapter just exposes the state
/// through the trait so `inferlet::sampler::weighted_sample` — which is
/// generic over `RngCore` — can drive it.
pub(crate) struct XorshiftRng {
    state: u64,
}

impl XorshiftRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { XORSHIFT_SEED_FALLBACK } else { seed },
        }
    }
}

impl RngCore for XorshiftRng {
    fn next_u32(&mut self) -> u32 {
        // High 32 bits of the 64-bit xorshift output — they mix faster than
        // the low bits for linear-feedback generators.
        (xorshift64_next(&mut self.state) >> 32) as u32
    }

    fn next_u64(&mut self) -> u64 {
        xorshift64_next(&mut self.state)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        let mut i = 0;
        while i < dest.len() {
            let bytes = self.next_u64().to_le_bytes();
            let take = (dest.len() - i).min(8);
            dest[i..i + take].copy_from_slice(&bytes[..take]);
            i += take;
        }
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> std::result::Result<(), rand_core::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

/// Options controlling the stochastic behavior of `ConstrainedSampler`.
///
/// `Default` disables every filter (rep_penalty=1.0, no top_p cutoff,
/// no top_k cap, fresh-entropy seed). Callers that want
/// `generation_config.json` parity with Qwen2.5 should pass
/// `{ rep_penalty: 1.05, top_p: Some(0.8), top_k: Some(20) }` — these
/// match vLLMs full sampling pipeline and keep the stochastic draw
/// from reaching into the long tail of the distribution (which is how
/// Phase D1 hallucinated a bogus integration ID).
#[derive(Debug, Clone, Copy)]
pub struct SamplerOptions {
    pub rep_penalty: f32,
    /// Nucleus (top-p) cutoff: keep the smallest set of allowed tokens
    /// whose cumulative probability is >= top_p, redistribute the rest.
    /// `None` or 1.0 disables filtering.
    pub top_p: Option<f32>,
    /// Keep only the `top_k` highest-probability allowed tokens.
    /// `None` disables filtering.
    pub top_k: Option<u32>,
    pub seed: Option<u64>,
}

impl Default for SamplerOptions {
    fn default() -> Self {
        Self {
            rep_penalty: 1.0,
            top_p: None,
            top_k: None,
            seed: None,
        }
    }
}

impl ConstrainedSampler {
    pub fn new(
        vocab: Vocab,
        special_tokens: Vocab,
        split_regex: String,
        grammar: String,
        eos_token_id: u32,
        escape_non_printable: bool,
    ) -> Self {
        Self::with_options(
            vocab,
            special_tokens,
            split_regex,
            grammar,
            eos_token_id,
            escape_non_printable,
            SamplerOptions::default(),
        )
    }

    /// Fully-specified constructor. Accepts rep_penalty and optional
    /// deterministic seed via `SamplerOptions`.
    pub fn with_options(
        vocab: Vocab,
        special_tokens: Vocab,
        split_regex: String,
        grammar: String,
        eos_token_id: u32,
        escape_non_printable: bool,
        opts: SamplerOptions,
    ) -> Self {
        let (ranks, words) = vocab;

        let rank_map: HashMap<u32, Vec<u8>> = ranks.into_iter().zip(words).collect();

        let (special_ranks, special_words) = special_tokens;
        let special_tokens: HashMap<String, u32> = special_words
            .into_iter()
            .map(|w| String::from_utf8(w).unwrap())
            .zip(special_ranks)
            .collect();

        let tokenizer = BytePairEncoder::new(
            rank_map,
            special_tokens,
            &split_regex,
            eos_token_id,
            escape_non_printable,
        );
        let tokenizer_env = tokenizer.unwrap().to_env();

        let grammar = TopLevelGrammar::from_lark(grammar);

        let factory = ParserFactory::new_simple(&tokenizer_env).unwrap();
        let parser = factory.create_parser(grammar);

        let constraint = Matcher::new(parser);
        let seed = opts
            .seed
            .filter(|s| *s != 0)
            .unwrap_or_else(initial_seed);
        let rep_penalty = if opts.rep_penalty.is_finite() && opts.rep_penalty > 0.0 {
            opts.rep_penalty
        } else {
            1.0
        };
        // Normalize top_p: treat None, NaN, or values >= 1.0 as disabled.
        // Clamp small values to a minimum so we dont end up with a
        // zero-sized allowed set after filtering.
        let top_p = match opts.top_p {
            Some(p) if p.is_finite() && p > 0.0 && p < 1.0 => p.max(0.01),
            _ => 1.0,
        };
        let top_k = opts.top_k.unwrap_or(0);
        ConstrainedSampler {
            inner: RefCell::new(Inner {
                constraint,
                eos_token_id,
                rng: XorshiftRng::new(seed),
                rep_penalty,
                top_p,
                top_k,
                history: EmittedHistory::new(DEFAULT_HISTORY_MAX),
            }),
        }
    }
}

impl Sample for ConstrainedSampler {
    fn sample(&self, token_ids: &[u32], probs: &[f32]) -> u32 {
        let mut inner = self.inner.borrow_mut();
        let res = match inner.constraint.compute_mask() {
            Ok(r) => r,
            Err(_) => return inner.eos_token_id,
        };
        if res.is_empty() {
            return inner.eos_token_id;
        }

        // Collect only grammar-allowed, positive-prob (id, prob) pairs; the
        // SDK primitives operate on the resulting sparse distribution.
        let mut dist: Vec<(u32, f32)> = token_ids
            .iter()
            .zip(probs.iter())
            .filter_map(|(&tid, &p)| (res.is_allowed(tid) && p > 0.0).then_some((tid, p)))
            .collect();

        // vLLM's full sampling pipeline in prob-space: rep_penalty -> sort ->
        // top_k -> top_p -> weighted draw. (See `inferlet::sampler::primitives`
        // for the "why prob-space, not logit-space" derivation — closing that
        // gap requires engine-side penalty application, tracked as F3 in
        // hc task #23.) Qwen2.5's generation_config.json ships
        // rep_penalty=1.05, top_k=20, top_p=0.8; without them the stochastic
        // draw reaches into the long tail and hallucinates (Phase D1).
        apply_repetition_penalty(&mut dist, &inner.history, inner.rep_penalty);
        dist.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        apply_top_k(&mut dist, inner.top_k);
        apply_top_p(&mut dist, inner.top_p);

        let sampled_token_id = if dist.is_empty() {
            // No allowed token with positive mass. Fall back to the first
            // token in the full-vocab grammar mask (same behavior as the
            // pre-refactor code).
            res.first_bit_set().unwrap_or(inner.eos_token_id as usize) as u32
        } else {
            weighted_sample(&dist, &mut inner.rng)
        };

        // Commit the chosen token to advance the parser state. Critical even
        // for fallback tokens — without it the grammar never advances and
        // every step returns the same fallback.
        let _ = inner.constraint.consume_token(sampled_token_id);

        // Track emitted token for future repetition-penalty lookups.
        inner.history.push(sampled_token_id);

        sampled_token_id
    }
}

// Byte-pair encoding adapted from tiktoken-rs:
// https://github.com/openai/tiktoken/blob/main/src/lib.rs

/// Performs the byte pair merging operation.
fn _byte_pair_merge(ranks: &HashMap<Vec<u8>, Rank>, piece: &[u8]) -> Vec<(usize, Rank)> {
    let mut parts = Vec::with_capacity(piece.len() + 1);

    // Creates initial parts and finds the first best merge.
    let mut min_rank: (Rank, usize) = (Rank::MAX, usize::MAX);
    for i in 0..piece.len() - 1 {
        let rank = *ranks.get(&piece[i..i + 2]).unwrap_or(&Rank::MAX);
        if rank < min_rank.0 {
            min_rank = (rank, i);
        }
        parts.push((i, rank));
    }
    parts.push((piece.len() - 1, Rank::MAX));
    parts.push((piece.len(), Rank::MAX));

    let get_rank = |parts: &Vec<(usize, Rank)>, i: usize| {
        if (i + 3) < parts.len() {
            *ranks
                .get(&piece[parts[i].0..parts[i + 3].0])
                .unwrap_or(&Rank::MAX)
        } else {
            Rank::MAX
        }
    };

    // Greedily merges the best pairs until no more merges are possible.
    while min_rank.0 != Rank::MAX {
        let i = min_rank.1;

        if i > 0 {
            parts[i - 1].1 = get_rank(&parts, i - 1);
        }
        parts[i].1 = get_rank(&parts, i);
        parts.remove(i + 1);

        min_rank = (Rank::MAX, usize::MAX);
        for (i, &(_, rank)) in parts[..parts.len() - 1].iter().enumerate() {
            if rank < min_rank.0 {
                min_rank = (rank, i);
            }
        }
    }
    parts
}

/// Encodes a byte slice into a sequence of token ranks using byte-pair encoding.
pub fn byte_pair_encode(piece: &[u8], ranks: &HashMap<Vec<u8>, Rank>) -> Vec<Rank> {
    if piece.len() == 1 {
        return vec![ranks[piece]];
    }
    _byte_pair_merge(ranks, piece)
        .windows(2)
        .map(|part| ranks[&piece[part[0].0..part[1].0]])
        .collect()
}

/// An encoder that uses byte-pair encoding to tokenize text.
#[derive(Clone)]
pub struct BytePairEncoder {
    encoder: HashMap<Vec<u8>, Rank>,
    special_tokens: HashSet<String>,
    special_tokens_encoder: HashMap<String, Rank>,
    regex: Regex,
    special_regex: Regex,
    tok_trie: TokTrie,
    escape_non_printable: bool,
}

impl BytePairEncoder {
    pub fn new(
        decoder: HashMap<Rank, Vec<u8>>,
        special_tokens_encoder: HashMap<String, Rank>,
        pattern: &str,
        eos_token: u32,
        escape_non_printable: bool,
    ) -> Result<Self> {
        let regex = Regex::new(pattern)?;
        let special_regex = {
            let parts = special_tokens_encoder
                .keys()
                .map(|s| fancy_regex::escape(s))
                .collect::<Vec<_>>();
            Regex::new(&parts.join("|"))?
        };

        let encoder: HashMap<Vec<u8>, Rank> =
            decoder.iter().map(|(k, v)| (v.clone(), *k)).collect();

        if encoder.len() != decoder.len() {
            bail!(
                "Encoder and decoder must be of equal length; maybe you had duplicate token indices in your encoder?"
            );
        }

        // Build the token vocabulary for TokTrie
        let mut max_rank = 0;
        for rank in decoder.keys() {
            max_rank = max_rank.max(*rank);
        }
        for rank in special_tokens_encoder.values() {
            max_rank = max_rank.max(*rank);
        }
        let n_vocab = (max_rank + 1) as usize;

        let mut tokens = vec![vec![]; n_vocab];

        for (rank, bytes) in decoder.iter() {
            // If escape_non_printable is enabled, the vocab contains escaped bytes
            // (e.g., "Ġ" for space). We need to unescape them for the TokTrie
            // so that the grammar parser sees raw bytes (e.g., 0x20 for space).
            let raw_bytes = if escape_non_printable {
                unescape_non_printable(bytes)
            } else {
                bytes.clone()
            };
            tokens[*rank as usize] = raw_bytes;
        }

        for (name, rank) in special_tokens_encoder.iter() {
            let mut spec_bytes = Vec::with_capacity(name.len() + 1);
            spec_bytes.push(TokTrie::SPECIAL_TOKEN_MARKER);
            spec_bytes.extend_from_slice(name.as_bytes());
            tokens[*rank as usize] = spec_bytes;
        }

        let special_tokens = special_tokens_encoder.keys().cloned().collect();

        // Fill gaps in the vocabulary with placeholder tokens
        for (i, token) in tokens.iter_mut().enumerate() {
            if token.is_empty() {
                let mut name = format!(".<[{i}]>").into_bytes();
                name[0] = TokTrie::SPECIAL_TOKEN_MARKER;
                *token = name;
            }
        }

        let tok_trie = TokTrie::from(
            &TokRxInfo {
                vocab_size: n_vocab as u32,
                tok_eos: eos_token,
                tok_end_of_turn: None,
                tok_unk: None,
                tok_pad: None,
                tok_bos: None,
            },
            &tokens,
        );

        Ok(Self {
            encoder,
            special_tokens,
            special_tokens_encoder,
            regex,
            special_regex,
            tok_trie,
            escape_non_printable,
        })
    }

    /// Encodes a string into tokens, respecting a set of allowed special tokens.
    pub fn encode(&self, text: &str) -> Vec<Rank> {
        let mut ret = vec![];

        let mut start = 0;
        loop {
            let mut next_special;
            let mut start_find = start;
            loop {
                next_special = self.special_regex.find_from_pos(text, start_find).unwrap();
                match next_special {
                    Some(m) => {
                        if self.special_tokens.contains(&text[m.start()..m.end()]) {
                            break;
                        }
                        start_find = m.start() + 1;
                    }
                    None => break,
                }
            }
            let end = next_special.map_or(text.len(), |m| m.start());

            for mat in self.regex.find_iter(&text[start..end]) {
                let mut piece = mat.unwrap().as_str().as_bytes();

                let escaped_piece = escape_non_printable(piece);
                if self.escape_non_printable {
                    piece = escaped_piece.as_bytes();
                }

                if let Some(token) = self.encoder.get(piece) {
                    ret.push(*token);
                    continue;
                }
                let tokens = byte_pair_encode(piece, &self.encoder);
                ret.extend(&tokens);
            }

            match next_special {
                // And here we push the special token
                Some(m) => {
                    let piece = m.as_str();
                    let token = self.special_tokens_encoder[piece];
                    ret.push(token);
                    start = m.end();
                }
                None => break,
            }
        }

        ret
    }

    /// Wraps the tokenizer in an `Arc` to create a `TokEnv`.
    pub fn to_env(self) -> TokEnv {
        Arc::new(self)
    }
}

impl TokenizerEnv for BytePairEncoder {
    fn tok_trie(&self) -> &TokTrie {
        &self.tok_trie
    }

    /// Tokenizes a byte slice, using BPE as a fallback for unknown sequences.
    fn tokenize_bytes(&self, s: &[u8]) -> Vec<TokenId> {
        self.tok_trie
            .tokenize_with_greedy_fallback(s, |s| self.encode(s))
    }

    /// Tokenizes a byte slice, handling special tokens correctly within fallback sequences.
    fn tokenize_bytes_special(&self, s: &[u8]) -> Vec<TokenId> {
        self.tok_trie.tokenize_with_greedy_fallback(s, |s| {
            self.tok_trie.tokenize_with_special(s, |s| self.encode(s))
        })
    }
}

/// Generate the 256-entry "byte-level" maps.
///
///  * `enc[byte] -> char`  (stage-2 encoding)
///  * `dec[char] -> byte`  (stage-2 decoding)
///
/// The algorithm is identical to OpenAI-tiktoken's `bytes_to_unicode()`.
///
/// Printable ranges kept as-is:
///   1.  '!' (0x21) .. '~' (0x7E)
///   2.  '¡' (0xA1) .. '¬' (0xAC)
///   3.  '®' (0xAE) .. 'ÿ' (0xFF)
///
/// Everything else (control bytes, space, TAB, …) is
/// remapped to the BMP starting at U+0100.
fn build_tables() -> ([char; 256], HashMap<char, u8>) {
    // Step 1: collect the "safe" byte values we keep unchanged
    let mut bs: Vec<u8> = (b'!'..=b'~').collect(); // 0x21–0x7E
    bs.extend(0xA1..=0xAC); // 0xA1–0xAC
    bs.extend(0xAE..=0xFF); // 0xAE–0xFF

    // cs will hold the *Unicode code points* corresponding to bs
    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();

    // Step 2: assign code points ≥ 0x100 to the remaining bytes
    let mut n = 0u32;
    for b in 0u8..=255 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n); // U+0100, U+0101, …
            n += 1;
        }
    }

    // Convert to char
    let cs: Vec<char> = cs.into_iter().map(|u| char::from_u32(u).unwrap()).collect();

    // Zip into the forward & reverse tables
    let mut enc = ['\0'; 256];
    let mut dec = HashMap::with_capacity(256);
    for (b, ch) in bs.into_iter().zip(cs.into_iter()) {
        enc[b as usize] = ch;
        dec.insert(ch, b);
    }
    (enc, dec)
}

/// Encode a byte slice with the Qwen/GPT byte-level mapping.
fn escape_non_printable(bytes: &[u8]) -> String {
    static TABLES: once_cell::sync::Lazy<([char; 256], HashMap<char, u8>)> =
        once_cell::sync::Lazy::new(build_tables);

    bytes.iter().map(|&b| TABLES.0[b as usize]).collect()
}

/// Decode an escaped string back to raw bytes (reverse of escape_non_printable).
/// This handles UTF-8 encoded strings where non-printable bytes were escaped to
/// Unicode code points >= U+0100.
fn unescape_non_printable(bytes: &[u8]) -> Vec<u8> {
    static TABLES: once_cell::sync::Lazy<([char; 256], HashMap<char, u8>)> =
        once_cell::sync::Lazy::new(build_tables);

    // Try to interpret as UTF-8 and decode each char
    match std::str::from_utf8(bytes) {
        Ok(s) => s
            .chars()
            .filter_map(|c| TABLES.1.get(&c).copied())
            .collect(),
        // If not valid UTF-8, return as-is (might be raw bytes already)
        Err(_) => bytes.to_vec(),
    }
}

#[cfg(test)]
mod rng_tests {
    //! Unit tests for the xorshift RNG backend + the XorshiftRng adapter
    //! that plumbs it into `rand_core::RngCore`. We intentionally do NOT
    //! exercise `ConstrainedSampler::sample()` directly — that requires a
    //! full tokenizer + grammar to construct a live `Matcher`. The math
    //! itself (rep_penalty / top_k / top_p / weighted_sample) is exhaustively
    //! covered by `inferlet::sampler::primitives`'s own unit tests; these
    //! tests pin only what is *new* and *pieclaw-local*: the RNG backend and
    //! its adapter.
    use super::*;

    #[test]
    fn xorshift_advances_and_is_deterministic_for_fixed_seed() {
        let mut s1 = 0xCAFEBABE_u64;
        let mut s2 = 0xCAFEBABE_u64;
        let a: Vec<u64> = (0..8).map(|_| xorshift64_next(&mut s1)).collect();
        let b: Vec<u64> = (0..8).map(|_| xorshift64_next(&mut s2)).collect();
        assert_eq!(a, b, "same seed must produce same sequence");
        // Zero-seed gets rescued to the fallback constant, not silently
        // stuck at zero.
        let mut z = 0u64;
        assert_ne!(xorshift64_next(&mut z), 0);
    }

    #[test]
    fn xorshift_rng_adapter_is_deterministic_for_fixed_seed() {
        let mut r1 = XorshiftRng::new(0x1234_5678);
        let mut r2 = XorshiftRng::new(0x1234_5678);
        let a: Vec<u32> = (0..16).map(|_| r1.next_u32()).collect();
        let b: Vec<u32> = (0..16).map(|_| r2.next_u32()).collect();
        assert_eq!(a, b, "same seed must produce same u32 sequence");
    }

    #[test]
    fn xorshift_rng_adapter_zero_seed_is_rescued() {
        // A zero seed must not stick the state at zero. The adapter's
        // constructor + the xorshift64* rescue branch cooperate to keep the
        // stream live.
        let mut r = XorshiftRng::new(0);
        let values: Vec<u32> = (0..4).map(|_| r.next_u32()).collect();
        assert!(
            values.iter().any(|&v| v != 0),
            "zero-seeded RNG produced all zeros"
        );
    }

    #[test]
    fn pipeline_with_xorshift_rng_terminates_and_respects_weights() {
        // Smoke-test the full `sample()` pipeline (minus the Matcher) with
        // the adapter-backed RNG. {10:0.9, 20:0.1} with no filtering — token
        // 10 should dominate over 10k draws.
        let mut rng = XorshiftRng::new(0xFEEDF00D);
        let mut count10 = 0usize;
        let n = 10_000;
        for _ in 0..n {
            let history = EmittedHistory::new(16);
            let mut dist: Vec<(u32, f32)> = vec![(10, 0.9), (20, 0.1)];
            apply_repetition_penalty(&mut dist, &history, 1.0);
            dist.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            apply_top_k(&mut dist, 0);
            apply_top_p(&mut dist, 1.0);
            if weighted_sample(&dist, &mut rng) == 10 {
                count10 += 1;
            }
        }
        let frac = count10 as f64 / n as f64;
        assert!(
            (0.86..0.94).contains(&frac),
            "expected ~0.9 for token 10 through XorshiftRng, got {frac}"
        );
    }
}
