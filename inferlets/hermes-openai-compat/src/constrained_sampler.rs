//! A custom sampler that constrains token generation to match a Lark grammar.
//!
//! Uses the `llguidance` library to compute token masks based on grammar state,
//! ensuring outputs are always syntactically valid.
//!
//! Ported from pie/sdk/examples/constrained-decoding/src/sampler.rs.

use fancy_regex::Regex;
use inferlet::sampler::Sample;
use inferlet::{Result, bail};
use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::{TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv};
use llguidance::{Matcher, ParserFactory};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

pub type Rank = u32;

pub struct ConstrainedSampler {
    inner: RefCell<Inner>,
}

struct Inner {
    constraint: Matcher,
    eos_token_id: u32,
    /// xorshift64* PRNG state. Seeded from WASI random (or host time outside
    /// WASM) at construction.
    rng_state: u64,
    /// Multiplicative divisor applied to the probability of any token
    /// present in `emitted_history` before the weighted draw. 1.0 disables
    /// the penalty (identity). Values >1 make repeats less likely.
    ///
    /// Matches vLLM's `repetition_penalty` semantics; Qwen2.5's
    /// `generation_config.json` ships `repetition_penalty=1.05`.
    rep_penalty: f32,
    /// Nucleus (top-p) cutoff, in (0.0, 1.0]. Values >= 1.0 disable.
    top_p: f32,
    /// Top-k cap. 0 disables.
    top_k: u32,
    /// Circular buffer of recently emitted token ids. Bounded to
    /// `history_max` to keep the per-step `.contains()` cheap.
    emitted_history: VecDeque<u32>,
    history_max: usize,
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

/// Uniform f32 in [0.0, 1.0). 24 bits of entropy (float mantissa precision).
#[inline]
fn next_uniform01(state: &mut u64) -> f32 {
    let bits = xorshift64_next(state) >> 40;
    (bits as f32) / ((1u64 << 24) as f32)
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
        let rng_state = opts
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
                rng_state,
                rep_penalty,
                top_p,
                top_k,
                emitted_history: VecDeque::with_capacity(DEFAULT_HISTORY_MAX),
                history_max: DEFAULT_HISTORY_MAX,
            }),
        }
    }
}

impl Sample for ConstrainedSampler {
    fn sample(&self, token_ids: &[u32], probs: &[f32]) -> u32 {
        let mut inner = self.inner.borrow_mut();
        let res = inner.constraint.compute_mask();

        if let Err(_) = res {
            return inner.eos_token_id;
        }

        let res = res.unwrap();

        if res.is_empty() {
            return inner.eos_token_id;
        }

        // Collect the (id, prob) pairs allowed by the grammar mask. We
        // then apply vLLMs full sampling pipeline before the weighted
        // draw: rep_penalty -> top_k -> top_p -> multinomial sample. This
        // matches Qwen2.5s generation_config.json (rep_penalty=1.05,
        // top_k=20, top_p=0.8) which vLLM auto-injects at model load;
        // without top_k/top_p the long tail of the distribution lets
        // stochastic sampling pick truly rare tokens and send the model
        // off-rails (Phase D1: hallucinated integration IDs).
        // rep_penalty: divide the prob of any token in the recent history
        // by `penalty`, matching `repetition_penalty` spiritually. NB: this
        // is an approximation — vLLMs formulation scales LOGITS
        // (`logit /= penalty` for positive logits) which is strictly
        // stronger for high-confidence tokens. We cannot replicate that
        // here because the engine returns post-softmax probs and the true
        // logit (which depends on the unknown normalization constant Z) is
        // not recoverable from probs alone. Closing that gap is F3 in
        // hc task #23 — it requires engine-side penalty application.
        //
        // Phase D1 evidence: this prob-space penalty + top_p/top_k
        // eliminates hallucinations (no more "1234567890" integration
        // IDs). But Phase D2 showed 2/3 trials still produce the specific
        // INT-004 alternation duplicate, which needs the stronger
        // logit-space effect.
        let rep_penalty = inner.rep_penalty;
        let penalize = rep_penalty > 1.0 && !inner.emitted_history.is_empty();
        let mut allowed: Vec<(u32, f32)> = Vec::with_capacity(token_ids.len());
        for (i, &tid) in token_ids.iter().enumerate() {
            if res.is_allowed(tid) {
                let mut p = probs[i];
                if penalize && inner.emitted_history.contains(&tid) {
                    p /= rep_penalty;
                }
                if p > 0.0 {
                    allowed.push((tid, p));
                }
            }
        }

        // top_k: keep the K highest-probability allowed tokens. 0 disables.
        // We sort the full allowed set descending so top_p can then walk
        // in-order. sort_unstable_by is fine because identical probs are
        // interchangeable for the downstream draw.
        if !allowed.is_empty() {
            allowed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let k = inner.top_k as usize;
            if k > 0 && allowed.len() > k {
                allowed.truncate(k);
            }
            // top_p: keep the shortest prefix whose cumulative mass >= top_p,
            // measured over the (already post-rep_penalty, post-top_k)
            // restricted distribution. This is the standard "nucleus" recipe;
            // we normalize the cumulative against `sum_kept` rather than
            // against the pre-filter total because rep_penalty has already
            // shifted mass around.
            let top_p = inner.top_p;
            if top_p < 1.0 {
                let sum: f32 = allowed.iter().map(|(_, p)| *p).sum();
                if sum > 0.0 {
                    let target = top_p * sum;
                    let mut acc = 0.0f32;
                    let mut keep = allowed.len();
                    for (i, (_, p)) in allowed.iter().enumerate() {
                        acc += *p;
                        if acc >= target {
                            keep = i + 1;
                            break;
                        }
                    }
                    if keep < allowed.len() {
                        allowed.truncate(keep);
                    }
                }
            }
        }
        let total: f32 = allowed.iter().map(|(_, p)| *p).sum();

        let sampled_token_id = if allowed.is_empty() || !total.is_finite() || total <= 0.0 {
            // No allowed token in the sparse distribution. Fall back to the
            // first token in the full-vocab mask (same behavior as before).
            res.first_bit_set().unwrap_or(inner.eos_token_id as usize) as u32
        } else if allowed.len() == 1 {
            allowed[0].0
        } else {
            let u = next_uniform01(&mut inner.rng_state) * total;
            let mut acc = 0.0f32;
            // Last-element default covers floating-point rounding where
            // `acc` never quite reaches `u`.
            let mut picked = allowed[allowed.len() - 1].0;
            for &(id, p) in &allowed {
                acc += p;
                if acc >= u {
                    picked = id;
                    break;
                }
            }
            picked
        };

        // Commit the chosen token to advance the parser state.
        // This is critical even for fallback tokens — without it the grammar
        // never advances and every step returns the same fallback.
        let _ = inner.constraint.consume_token(sampled_token_id);

        // Track emitted token for future repetition-penalty lookups. Bound
        // the window so `.contains()` stays cheap and memory is fixed.
        if inner.history_max > 0 {
            if inner.emitted_history.len() == inner.history_max {
                inner.emitted_history.pop_front();
            }
            inner.emitted_history.push_back(sampled_token_id);
        }

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
    //! Unit tests for the xorshift RNG and the weighted-draw primitive
    //! used by `ConstrainedSampler::sample`. These tests intentionally do
    //! NOT exercise the `Matcher`/llguidance path — that requires a full
    //! tokenizer + grammar. They pin the weighted-draw math itself so a
    //! future refactor cannot silently regress to argmax.
    use super::*;

    /// Port of the weighted-sample math from `sample()` so we can test it
    /// without a live `Matcher`.
    fn weighted_draw(state: &mut u64, allowed: &[(u32, f32)]) -> u32 {
        assert!(!allowed.is_empty());
        let total: f32 = allowed.iter().map(|(_, p)| *p).sum();
        let u = next_uniform01(state) * total;
        let mut acc = 0.0f32;
        let mut picked = allowed[allowed.len() - 1].0;
        for &(id, p) in allowed {
            acc += p;
            if acc >= u {
                picked = id;
                break;
            }
        }
        picked
    }

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
    fn uniform01_stays_in_range() {
        let mut s = 0xDEADBEEF_u64;
        for _ in 0..1024 {
            let u = next_uniform01(&mut s);
            assert!((0.0..1.0).contains(&u), "uniform out of range: {u}");
        }
    }

    #[test]
    fn weighted_draw_follows_weights() {
        // {A:0.7, B:0.3} over 50k trials should land within 2% of 70/30.
        let allowed = vec![(10u32, 0.7), (20u32, 0.3)];
        let mut s = 0x12345678_u64;
        let n = 50_000;
        let mut count_a = 0;
        for _ in 0..n {
            if weighted_draw(&mut s, &allowed) == 10 {
                count_a += 1;
            }
        }
        let ratio = count_a as f32 / n as f32;
        assert!(
            (ratio - 0.7).abs() < 0.02,
            "expected ~0.70 A, got {ratio} ({count_a}/{n})"
        );
    }

    #[test]
    fn weighted_draw_non_degenerate_for_close_probs() {
        // Argmax would pick A every time. Weighted draw must produce both.
        let allowed = vec![(1u32, 0.51), (2u32, 0.49)];
        let mut s = 0x1_u64;
        let mut seen_a = false;
        let mut seen_b = false;
        for _ in 0..200 {
            match weighted_draw(&mut s, &allowed) {
                1 => seen_a = true,
                2 => seen_b = true,
                _ => unreachable!(),
            }
            if seen_a && seen_b {
                break;
            }
        }
        assert!(seen_a && seen_b, "weighted draw was effectively greedy");
    }

    #[test]
    fn weighted_draw_single_id_always_picks_it() {
        let allowed = vec![(42u32, 0.33)];
        let mut s = 0xABCD_u64;
        for _ in 0..32 {
            assert_eq!(weighted_draw(&mut s, &allowed), 42);
        }
    }

    /// Port of the penalty application from `sample()` — prob-space
    /// `p /= penalty` for tokens in history. This is an approximation of
    /// vLLMs logit-space formula; see the long comment in `sample()`.
    fn apply_rep_penalty(
        allowed: &[(u32, f32)],
        history: &[u32],
        penalty: f32,
    ) -> Vec<(u32, f32)> {
        if penalty <= 1.0 || history.is_empty() {
            return allowed.to_vec();
        }
        let hist_set: HashSet<u32> = history.iter().copied().collect();
        allowed
            .iter()
            .map(|&(id, p)| {
                let q = if hist_set.contains(&id) { p / penalty } else { p };
                (id, q)
            })
            .collect()
    }

    #[test]
    fn rep_penalty_shifts_mass_away_from_recent_tokens() {
        // Uniform {A:0.5, B:0.5}, A in history, penalty=1.05. A->0.476
        // after /=1.05, renorm total = 0.976, new B share = 0.5/0.976 =
        // 0.512. Expect B ~51% in 50k trials.
        let allowed = vec![(100u32, 0.5), (200u32, 0.5)];
        let history = vec![100u32];
        let penalized = apply_rep_penalty(&allowed, &history, 1.05);
        let mut s = 0xFEED_u64;
        let n = 50_000;
        let mut count_b = 0;
        for _ in 0..n {
            if weighted_draw(&mut s, &penalized) == 200 {
                count_b += 1;
            }
        }
        let ratio = count_b as f32 / n as f32;
        assert!(
            (0.50..=0.53).contains(&ratio),
            "expected B ratio ~0.512 with rep_penalty=1.05 (prob-space), got {ratio}"
        );
    }

    #[test]
    fn rep_penalty_disabled_when_penalty_is_one() {
        let allowed = vec![(1u32, 0.5), (2u32, 0.5)];
        let history = vec![1u32, 1u32, 1u32];
        let penalized = apply_rep_penalty(&allowed, &history, 1.0);
        assert_eq!(penalized, allowed);
    }

    /// Replicates the top_k/top_p filtering path from `sample()` so we
    /// can test it without a live Matcher. Mirrors the order in sample():
    /// sort desc -> truncate top_k -> truncate at top_p cumulative mass.
    fn filter_top_k_top_p(
        mut allowed: Vec<(u32, f32)>,
        top_k: usize,
        top_p: f32,
    ) -> Vec<(u32, f32)> {
        allowed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if top_k > 0 && allowed.len() > top_k {
            allowed.truncate(top_k);
        }
        if top_p < 1.0 {
            let sum: f32 = allowed.iter().map(|(_, p)| *p).sum();
            if sum > 0.0 {
                let target = top_p * sum;
                let mut acc = 0.0;
                let mut keep = allowed.len();
                for (i, (_, p)) in allowed.iter().enumerate() {
                    acc += *p;
                    if acc >= target {
                        keep = i + 1;
                        break;
                    }
                }
                if keep < allowed.len() {
                    allowed.truncate(keep);
                }
            }
        }
        allowed
    }

    #[test]
    fn top_k_keeps_highest_probability_tokens() {
        let allowed = vec![(1, 0.1), (2, 0.5), (3, 0.3), (4, 0.08), (5, 0.02)];
        let kept = filter_top_k_top_p(allowed, 2, 1.0);
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0].0, 2); // 0.5
        assert_eq!(kept[1].0, 3); // 0.3
    }

    #[test]
    fn top_k_zero_disables_filter() {
        let allowed = vec![(1, 0.5), (2, 0.5)];
        let kept = filter_top_k_top_p(allowed, 0, 1.0);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn top_p_keeps_shortest_nucleus() {
        // Mass: 0.5 + 0.3 + 0.1 + 0.08 + 0.02 = 1.0. top_p=0.8 needs
        // cumulative >= 0.8, so after sorting desc we keep {0.5, 0.3}.
        let allowed = vec![(1, 0.1), (2, 0.5), (3, 0.3), (4, 0.08), (5, 0.02)];
        let kept = filter_top_k_top_p(allowed, 0, 0.8);
        assert_eq!(kept.len(), 2, "nucleus should be {{0.5, 0.3}} for top_p=0.8");
        assert_eq!(kept[0].0, 2);
        assert_eq!(kept[1].0, 3);
    }

    #[test]
    fn top_p_one_disables_filter() {
        let allowed = vec![(1, 0.5), (2, 0.3), (3, 0.2)];
        let kept = filter_top_k_top_p(allowed.clone(), 0, 1.0);
        assert_eq!(kept.len(), 3);
    }

    #[test]
    fn top_k_then_top_p_composed() {
        // 10 tokens; top_k=5 keeps top 5 by mass, then top_p=0.7 cuts
        // further. Sorted probs: 0.35, 0.25, 0.15, 0.10, 0.08, ... =
        // sum_kept=0.93; target=0.7*0.93=0.651. Cumulative: 0.35, 0.60,
        // 0.75 >= 0.651 -> keep 3.
        let allowed = vec![
            (1, 0.02), (2, 0.35), (3, 0.01), (4, 0.15), (5, 0.25),
            (6, 0.08), (7, 0.10), (8, 0.01), (9, 0.02), (10, 0.01),
        ];
        let kept = filter_top_k_top_p(allowed, 5, 0.7);
        assert_eq!(kept.len(), 3, "expected 3 tokens after top_k=5 then top_p=0.7");
        assert_eq!(kept[0].0, 2); // 0.35
        assert_eq!(kept[1].0, 5); // 0.25
        assert_eq!(kept[2].0, 4); // 0.15
    }

    #[test]
    fn top_p_never_returns_empty_when_input_nonempty() {
        // Even with a very small top_p, at least one token must survive
        // (the highest-prob one always satisfies cumulative >= target on
        // its own).
        let allowed = vec![(1, 0.9), (2, 0.05), (3, 0.05)];
        for p in [0.01_f32, 0.1, 0.5, 0.99] {
            let kept = filter_top_k_top_p(allowed.clone(), 0, p);
            assert!(!kept.is_empty(), "top_p={p} produced empty set");
        }
    }

    #[test]
    fn rep_penalty_pushes_toward_alternatives_on_heavy_repeat() {
        // Simulate the INT-004 case: one token dominates (0.8) and is in
        // history. With penalty=1.5: 0.8/1.5 = 0.533. Renormalized total =
        // 0.733. Alt share = 0.2 / 0.733 = 0.273.
        let allowed = vec![(77u32, 0.8), (99u32, 0.2)];
        let history = vec![77u32];
        let penalized = apply_rep_penalty(&allowed, &history, 1.5);
        let mut s = 0xC0DE_u64;
        let n = 20_000;
        let mut count_alt = 0;
        for _ in 0..n {
            if weighted_draw(&mut s, &penalized) == 99 {
                count_alt += 1;
            }
        }
        let ratio = count_alt as f32 / n as f32;
        assert!(
            (0.22..=0.33).contains(&ratio),
            "expected alt ratio ~0.27 with prob-space penalty=1.5, got {ratio}"
        );
    }
}
