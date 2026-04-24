//! Regression guard for VENDOR_SOURCE.md divergence #5.
//!
//! Greps src/handler.rs and src/variant.rs for `eprintln!` / `println!` / writes
//! to `std::io::stderr()` that occur BEFORE the first `.await` inside an async
//! function, AND anywhere inside the synchronous helper `save_session_kv_state`.
//! Any such call hangs the request under wstd on wasm32-wasip2.
//!
//! History: previously only the sync-prefix-of-async-fn case was guarded,
//! assuming that sync helpers called from async context were safe if they
//! did not contain `.await`. Task #47's N≥8 IncompleteMessage wedge proved
//! that assumption wrong: `save_session_kv_state` is `fn`, but it is invoked
//! inside an async HTTP handler after many awaits, and an `eprintln!` hit
//! on the error path wedged 12 of 16 concurrent requests for the full 300s
//! client timeout. The export-error eprintln has since been deleted; this
//! guard keeps it that way.

use std::fs;

const HANDLER: &str = "src/handler.rs";
const VARIANT: &str = "src/variant.rs";

fn forbidden_call(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with("eprintln!") || t.starts_with("println!")
        || t.contains("std::io::stderr()") || t.contains("std::io::stdout()")
}

#[test]
fn no_sync_eprintln_in_async_handlers() {
    let root = env!("CARGO_MANIFEST_DIR");
    for relpath in [HANDLER, VARIANT] {
        let path = format!("{root}/{relpath}");
        let src = fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("could not read {}", path));
        let mut in_async = false;
        let mut saw_await = false;
        // Separate flag for the known-hazard sync helper called from async
        // context (save_session_kv_state). Task #47 proved ANY eprintln in
        // this function wedges concurrent requests under wstd — the `fn` vs
        // `async fn` distinction doesn't save us.
        let mut in_save_session = false;
        for (i, line) in src.lines().enumerate() {
            let lineno = (i as u32) + 1;
            let t = line.trim_start();
            if t.starts_with("pub async fn") || t.starts_with("async fn") {
                in_async = true;
                saw_await = false;
                continue;
            }
            if t.starts_with("fn save_session_kv_state") {
                in_save_session = true;
                continue;
            }
            // Coarse: a closing brace at column 0 ends the function.
            if line.starts_with('}') {
                in_async = false;
                saw_await = false;
                in_save_session = false;
                continue;
            }
            if in_save_session && forbidden_call(line) {
                panic!("{}:{}: forbidden log inside save_session_kv_state — this sync helper is called from async context and any eprintln wedges concurrent requests (task #47)",
                    relpath, lineno);
            }
            if !in_async { continue; }
            if t.contains(".await") { saw_await = true; }
            if !saw_await && forbidden_call(line) {
                panic!("{}:{}: forbidden sync-prefix log in async fn (see VENDOR_SOURCE.md #5)",
                    relpath, lineno);
            }
        }
    }
}
