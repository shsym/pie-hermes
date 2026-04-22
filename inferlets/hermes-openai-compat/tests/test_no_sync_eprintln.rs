//! Regression guard for VENDOR_SOURCE.md divergence #5.
//!
//! Greps src/handler.rs and src/variant.rs for `eprintln!` / `println!` / writes
//! to `std::io::stderr()` that occur BEFORE the first `.await` inside an async
//! function. Any new such call hangs the request under wstd on wasm32-wasip2.
//!
//! The test allowlists the two pre-existing inherited calls VENDOR_SOURCE.md #5
//! calls out. Both fire only behind an `.await` (so the test's `saw_await` gate
//! also skips them) but they are listed here as belt-and-suspenders documentation
//! so future readers can see them enumerated alongside the rule.
//!
//! Allowlist line lookup (verified 2026-04-22 against current src/handler.rs):
//!   * line 431: constrained-sampler fallback `eprintln!` inside
//!     `async fn build_sampler` (post-`.await` at line 427).
//!   * line 1967: `export_kv_pages_sync` failure `eprintln!` inside the
//!     synchronous helper `fn save_session_kv_state` (not async, so the
//!     in_async gate also skips it; listed for completeness).
//! VENDOR_SOURCE.md #5 referenced these as `~430` and `~1965` — drift to
//! 431 / 1967 is from intervening edits, not new violations.

use std::fs;

const HANDLER: &str = "src/handler.rs";
const VARIANT: &str = "src/variant.rs";
const ALLOWLIST: &[(&str, u32)] = &[
    // (file, line) of inherited error-branch calls behind an .await
    (HANDLER, 431),
    (HANDLER, 1967),
];

fn forbidden_call(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with("eprintln!") || t.starts_with("println!")
        || t.contains("std::io::stderr()") || t.contains("std::io::stdout()")
}

#[test]
fn no_sync_eprintln_in_async_handlers() {
    for path in [HANDLER, VARIANT] {
        let src = fs::read_to_string(path)
            .unwrap_or_else(|_| panic!("could not read {}", path));
        let mut in_async = false;
        let mut saw_await = false;
        for (i, line) in src.lines().enumerate() {
            let lineno = (i as u32) + 1;
            let t = line.trim_start();
            if t.starts_with("pub async fn") || t.starts_with("async fn") {
                in_async = true;
                saw_await = false;
                continue;
            }
            // Coarse: a closing brace at column 0 ends the function.
            if line.starts_with('}') {
                in_async = false;
                saw_await = false;
                continue;
            }
            if !in_async { continue; }
            if t.contains(".await") { saw_await = true; }
            if !saw_await && forbidden_call(line) {
                let allowed = ALLOWLIST.iter().any(|(f, l)| *f == path && *l == lineno);
                assert!(allowed,
                    "{}:{}: forbidden sync-prefix log in async fn (see VENDOR_SOURCE.md #5)",
                    path, lineno);
            }
        }
    }
}
