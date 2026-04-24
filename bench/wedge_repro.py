#!/usr/bin/env python3
"""Minimal reproduction for pie-hermes hyper::Error(IncompleteMessage) at N>=8.

Three arms, each fires N concurrent chat-completion requests and reports
which req_ids completed vs wedged.  Run each arm against the pie endpoint
AFTER a warmup has loaded the shared prefix into block_cache.

Arms:
  auto     — shared bearer + shared system prompt (auto-derives same session_id for all N)
  unique   — shared bearer + per-request distinct X-Pie-Session-Id header
  nobearer — no Authorization header at all (falls through to direct path, no session)

Usage:
    python3 wedge_repro.py --url http://127.0.0.1:18001 --model Qwen/Qwen2.5-72B-Instruct-AWQ --n 16 --arm auto
"""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import http.client
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

SYS_PROMPT = (
    "You are a precise, terse technical assistant.  Answer in one short sentence. "
    "Do not add preamble, apologies, or conversational filler."
)

QUERIES = [
    "What is 17 times 23?",
    "Capital of Uruguay?",
    "Time complexity of merge sort?",
    "Who wrote The Brothers Karamazov?",
    "Smallest prime larger than 100?",
    "Cosine similarity formula?",
    "Year of Chernobyl disaster?",
    "Atomic number of carbon?",
    "List vs tuple in Python?",
    "Three sorting algorithms O(n log n)?",
    "Derivative of x^3 + 2x?",
    "Country with highest GDP per capita 2024?",
    "HTTP 418 meaning?",
    "Nitrogen boiling point at 1 atm?",
    "Process vs thread?",
    "Who proved Fermat's Last Theorem?",
]


def one(url: str, body: dict, headers: dict, req_id: str, timeout: float) -> dict:
    t0 = time.perf_counter()
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        t1 = time.perf_counter()
        usage = data.get("usage") or {}
        pie = data.get("pie_cache") or {}
        return {
            "req_id": req_id,
            "ok": True,
            "e2e_ms": 1000 * (t1 - t0),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "prefill_skipped": pie.get("prefill_tokens_skipped"),
            "mode": pie.get("mode"),
            "fallback_reason": pie.get("fallback_reason"),
        }
    except Exception as e:
        t1 = time.perf_counter()
        return {
            "req_id": req_id,
            "ok": False,
            "e2e_ms": 1000 * (t1 - t0),
            "error": str(e)[:200],
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--arm", choices=["auto", "unique", "nobearer"], required=True)
    ap.add_argument("--bearer", default="task2296-repro")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--out", help="Write JSON result file")
    args = ap.parse_args()

    # Warmup once
    print(f"[warmup] POST /v1/pie/prompt-cache/chat-prefix:warm", flush=True)
    warmup_body = {
        "model": args.model,
        "messages": [{"role": "system", "content": SYS_PROMPT}],
    }
    warmup_headers = {}
    if args.arm != "nobearer":
        warmup_headers["Authorization"] = f"Bearer {args.bearer}"
    try:
        req = urllib.request.Request(
            args.url.rstrip("/") + "/v1/pie/prompt-cache/chat-prefix:warm",
            data=json.dumps(warmup_body).encode(),
            headers={"Content-Type": "application/json", **warmup_headers},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120.0) as resp:
            print(f"[warmup] ok: {resp.read()[:200]}", flush=True)
    except Exception as e:
        print(f"[warmup] failed (continuing): {e}", flush=True)

    # Build per-arm request parameters
    results = []
    print(f"[run] arm={args.arm} n={args.n} timeout={args.timeout}s", flush=True)
    t_start = time.perf_counter()

    def make_request(i: int) -> dict:
        body = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": QUERIES[i % len(QUERIES)]},
            ],
            "temperature": 0.0,
            "seed": i,
            "max_tokens": 16,
            "stream": False,
        }
        headers = {}
        if args.arm == "auto":
            headers["Authorization"] = f"Bearer {args.bearer}"
        elif args.arm == "unique":
            headers["Authorization"] = f"Bearer {args.bearer}"
            headers["X-Pie-Session-Id"] = f"repro-{uuid.uuid4()}"
        # nobearer: no Authorization header at all
        return one(args.url, body, headers, f"r{i:03d}", args.timeout)

    with cf.ThreadPoolExecutor(max_workers=args.n) as pool:
        futures = [pool.submit(make_request, i) for i in range(args.n)]
        for f in cf.as_completed(futures):
            r = f.result()
            tag = "OK " if r["ok"] else "ERR"
            detail = (
                f"e2e={r['e2e_ms']:.0f}ms skip={r.get('prefill_skipped')} mode={r.get('mode')}"
                if r["ok"] else f"err={r.get('error')}"
            )
            print(f"  [{r['req_id']}] {tag} {detail}", flush=True)
            results.append(r)

    t_end = time.perf_counter()
    window = t_end - t_start
    ok = sum(1 for r in results if r["ok"])
    err = len(results) - ok
    print(f"\n[summary] arm={args.arm} n={args.n} ok={ok} err={err} window={window:.1f}s", flush=True)
    if ok:
        e2e_ok = sorted(r["e2e_ms"] for r in results if r["ok"])
        print(f"  e2e ok (ms): min={e2e_ok[0]:.0f} p50={e2e_ok[len(e2e_ok)//2]:.0f} max={e2e_ok[-1]:.0f}")
    if err:
        errors = {}
        for r in results:
            if not r["ok"]:
                k = r.get("error", "?")[:80]
                errors[k] = errors.get(k, 0) + 1
        print(f"  errors: {errors}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "arm": args.arm,
                "n": args.n,
                "window_sec": window,
                "ok_count": ok,
                "err_count": err,
                "results": results,
            }, f, indent=2)
        print(f"  saved {args.out}")


if __name__ == "__main__":
    main()
