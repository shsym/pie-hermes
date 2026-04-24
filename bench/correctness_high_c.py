#!/usr/bin/env python3
"""Correctness harness for pie-hermes under high concurrency + memory pressure.

Complements bench/mempressure.py (which checks throughput only) by actually
diffing response CONTENT across stacks and across concurrency levels.

Protocol:
  phase=capture --stack <stockvllm|pie> --n <c> --scenario <s1|s2>
    Fires `c` concurrent identical-seed requests per query, saves
    (scenario, n, stack, query_id, slot, content, completion_tokens) rows.
    With temp=0 + fixed seed, all slots of the same query should produce
    the same content; that's the intra-run determinism check.

  phase=diff --golden <capture.json> --candidate <capture.json>
    Compares two capture files sharing the same scenario+n. Reports:
      - exact_match: content string-equal
      - empty: candidate content is "" (the N=32 anomaly signature)
      - prefix_match: first 32 chars match (soft)
      - mismatch: neither
    Also checks slot-determinism inside each file: did slots 0..n-1 of
    query q all produce the same content?

Usage:
  python3 correctness_high_c.py capture --stack stockvllm --n 16 --scenario s1 \\
      --base-url http://127.0.0.1:9090 --out /tmp/gold-stockvllm-s1-n16.json
  python3 correctness_high_c.py capture --stack pie --n 16 --scenario s1 \\
      --base-url http://127.0.0.1:9090 --bearer tt47 --out /tmp/pie-s1-n16.json
  python3 correctness_high_c.py diff --golden /tmp/gold-stockvllm-s1-n16.json \\
      --candidate /tmp/pie-s1-n16.json
"""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SYS_PROMPT = (
    "You are Hermes, a precise technical assistant.  Answer the question directly in at most "
    "one short sentence.  No preamble, no filler, no apologies."
)

S1_QUERIES = [
    ("q01", "What is 17 times 23?"),
    ("q02", "Name the capital of Uruguay."),
    ("q03", "What is the time complexity of merge sort?"),
    ("q04", "Who wrote The Brothers Karamazov?"),
    ("q05", "What is the smallest prime larger than 100?"),
    ("q06", "What is the atomic number of carbon?"),
    ("q07", "In which year did the Chernobyl disaster occur?"),
    ("q08", "What is the derivative of x^3 + 2x with respect to x?"),
    ("q09", "What does HTTP status code 418 mean?"),
    ("q10", "Who proved Fermat's Last Theorem?"),
]

# S2 shares a ~1.5k-token document so each request has both a shared
# system prefix AND a large shared document section — maximal KV reuse
# AND higher per-request KV footprint → approaches real memory pressure
# at high N on an A100-80GB pool.
S2_DOCUMENT = (
    "Storage Tier ST3 Operations Memo, revision 2026-03-12.\n\n"
    "ST3 is the hot storage tier backing all OLTP workloads. It is a "
    "three-AZ replicated NVMe cluster with a per-shard leader that "
    "handles writes and reads for a primary key range of 2^20 keys. "
    "Reads follow the leader unless the leader is quarantined, in "
    "which case they fan out to a majority quorum of followers. "
    "Followers apply the replication log with a bounded lag budget "
    "of 250ms; exceeding this budget marks the follower unhealthy "
    "and the leader routes reads to the remaining quorum. "
    "Write amplification is capped at 3x via a coalescing write buffer "
    "that batches updates in 4 KiB pages and flushes on a 5 ms timer "
    "or when the buffer reaches 64 KiB.\n\n"
    "Failover: the leader publishes a heartbeat to the shard gossip "
    "channel every 750 ms. A peer that fails to observe three "
    "consecutive heartbeats initiates a leader election. The election "
    "quorum is 2 of 3 AZs. The election winner acquires a fenced "
    "epoch counter; all subsequent writes are tagged with the epoch "
    "so followers can reject stale writes from a deposed leader that "
    "hasn't noticed its quarantine yet. "
    "Clients cache leader identity for 30 seconds and refresh via a "
    "GET /shards/<id>/leader call on epoch mismatch. The p99 leader-"
    "refresh call is 8 ms.\n\n"
    "Backup: snapshots run hourly on one follower per shard, "
    "rotating through the three AZs to avoid hot-spotting a single "
    "datacenter. Snapshots are encrypted with a per-shard data "
    "encryption key wrapped by the account KMS key. Snapshot storage "
    "lives in a separate bucket in another region; restore SLA is 15 "
    "minutes for shards up to 1 TiB.\n\n"
    "Compaction: major compaction runs on the leader during the "
    "shard's low-traffic window (configurable, default 02:00-06:00 "
    "local). During major compaction, write throughput is capped at "
    "50% of the normal budget and p99 write latency rises from 2 ms "
    "to 15 ms. Minor compaction runs continuously with negligible "
    "latency impact. Compaction uses a leveled strategy with 8 "
    "levels; level 0 holds raw write batches, level 7 holds "
    "fully-compacted data pages with bloom filters.\n\n"
    "Observability: every request emits a trace span to the shared "
    "OTEL collector with the shard id, leader epoch, request kind, "
    "and outcome. Prometheus metrics include p50/p99 latency by "
    "request kind, write buffer occupancy, replication lag per "
    "follower, and compaction progress. Alerts fire on replication "
    "lag > 500 ms sustained for 1 minute, or write p99 > 30 ms for 5 "
    "minutes outside the low-traffic window.\n\n"
    "Client guidance: target the leader directly when the key range "
    "is known. Use the metadata service only for cold-start discovery "
    "or on epoch-mismatch retries. Bound client-side retry to 3 "
    "attempts with exponential backoff capped at 200 ms; retries "
    "beyond that are counterproductive and will be rejected by the "
    "leader's admission controller.\n\n"
    "Quotas: each account gets 10K writes/sec per shard and 50K "
    "reads/sec per shard by default. Exceeding the quota returns "
    "HTTP 429 with a Retry-After header. Upgrading the quota "
    "requires a support ticket; same-day approval is available "
    "for enterprise accounts. Quota increases take effect "
    "within 5 minutes.\n"
)

S2_QUERIES = [
    ("q01", "What is the ST3 replication lag budget?"),
    ("q02", "What fraction of normal write throughput is allowed during major compaction?"),
    ("q03", "How long does a client cache leader identity?"),
    ("q04", "What is the election quorum requirement?"),
    ("q05", "What HTTP status code is returned when quotas are exceeded?"),
    ("q06", "What is the maximum number of retries recommended for clients?"),
    ("q07", "How many levels does leveled compaction have?"),
    ("q08", "What is the heartbeat interval?"),
    ("q09", "How often do snapshots run?"),
    ("q10", "What is the restore SLA for shards up to 1 TiB?"),
]


def _http_post_json(base_url, path, body, headers, timeout):
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def build_body(scenario, model, qid, qtext, seed, max_tokens):
    if scenario == "s1":
        messages = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": qtext},
        ]
    else:  # s2
        messages = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": S2_DOCUMENT + "\n\nQUESTION: " + qtext},
        ]
    return {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "seed": seed,
        "max_tokens": max_tokens,
        "stream": False,
    }


def fire_one(base_url, body, headers, timeout, meta, unique_session=False):
    if unique_session:
        import uuid
        headers = {**headers, "X-Pie-Session-Id": f"corr-{meta['query_id']}-{meta['slot']}-{uuid.uuid4().hex[:8]}"}
    return _fire_one_core(base_url, body, headers, timeout, meta)


def _fire_one_core(base_url, body, headers, timeout, meta):
    t0 = time.perf_counter()
    try:
        resp = _http_post_json(base_url, "/v1/chat/completions", body, headers, timeout)
    except Exception as e:
        return {**meta, "ok": False, "error": str(e)[:200],
                "e2e_ms": 1000 * (time.perf_counter() - t0)}
    t1 = time.perf_counter()
    content = ""
    finish = None
    try:
        choice = resp["choices"][0]
        content = choice["message"].get("content") or ""
        finish = choice.get("finish_reason")
    except Exception:
        content = ""
    usage = resp.get("usage") or {}
    pie = resp.get("pie_cache") or {}
    return {
        **meta,
        "ok": True,
        "e2e_ms": 1000 * (t1 - t0),
        "content": content,
        "finish_reason": finish,
        "completion_tokens": usage.get("completion_tokens"),
        "prefill_skipped": pie.get("prefill_tokens_skipped"),
    }


def cmd_capture(args):
    queries = S1_QUERIES if args.scenario == "s1" else S2_QUERIES
    headers = {}
    if args.stack == "pie" and args.bearer:
        headers["Authorization"] = f"Bearer {args.bearer}"

    # Warmup: single request establishes prefix cache
    print(f"[warmup] pie={args.stack=='pie'}", flush=True)
    try:
        if args.stack == "pie":
            _http_post_json(args.base_url, "/v1/pie/prompt-cache/chat-prefix:warm",
                            {"model": args.model,
                             "messages": [{"role": "system", "content": SYS_PROMPT}]},
                            headers, timeout=120.0)
        # also a plain warmup completion
        _http_post_json(args.base_url, "/v1/chat/completions",
                        build_body(args.scenario, args.model, "warm", "What is 1+1?", 0, 4),
                        headers, timeout=120.0)
    except Exception as e:
        print(f"[warmup] failed (continuing): {e}", flush=True)

    rows = []
    n = args.n
    print(f"[capture] stack={args.stack} scenario={args.scenario} n={n} queries={len(queries)}", flush=True)
    for qid, qtext in queries:
        body = build_body(args.scenario, args.model, qid, qtext, args.seed, args.max_tokens)
        submits = []
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(max_workers=n) as pool:
            for slot in range(n):
                meta = {"scenario": args.scenario, "n": n, "stack": args.stack,
                        "query_id": qid, "slot": slot, "seed": args.seed}
                submits.append(pool.submit(fire_one, args.base_url, body, headers,
                                           args.timeout, meta, args.unique_session))
            for f in cf.as_completed(submits):
                rows.append(f.result())
        dt = time.perf_counter() - t0
        contents = sorted({r.get("content", "") for r in rows if r.get("query_id") == qid and r.get("ok")})
        empties = sum(1 for r in rows if r.get("query_id") == qid and r.get("ok") and r.get("content") == "")
        errs = sum(1 for r in rows if r.get("query_id") == qid and not r.get("ok"))
        sample = (contents[0] if contents else "")[:80]
        print(f"  [{qid}] dt={dt:.1f}s ok={n-errs}/{n} empty={empties} distinct_contents={len(contents)} sample={sample!r}", flush=True)

    Path(args.out).write_text(json.dumps({
        "scenario": args.scenario, "n": n, "stack": args.stack,
        "seed": args.seed, "max_tokens": args.max_tokens,
        "model": args.model, "rows": rows,
    }, indent=2))
    print(f"[capture] saved {args.out} ({len(rows)} rows)", flush=True)


def cmd_diff(args):
    gold = json.loads(Path(args.golden).read_text())
    cand = json.loads(Path(args.candidate).read_text())
    assert gold["scenario"] == cand["scenario"], "scenario mismatch"
    scenario = gold["scenario"]
    # collapse gold: (scenario, query_id) -> canonical content (if slots agree)
    def build_canonical(capture):
        by_q = {}
        for r in capture["rows"]:
            if not r.get("ok"):
                continue
            by_q.setdefault(r["query_id"], []).append(r.get("content", ""))
        canon = {}
        for q, contents in by_q.items():
            distinct = list({c for c in contents})
            if len(distinct) == 1:
                canon[q] = {"content": distinct[0], "determinism": "unanimous", "slots": len(contents)}
            else:
                # mode (most common)
                from collections import Counter
                mc = Counter(contents).most_common()
                canon[q] = {"content": mc[0][0], "determinism": f"split {dict(Counter(contents))}",
                            "slots": len(contents), "all_distinct": distinct}
        return canon

    gc = build_canonical(gold)
    cc = build_canonical(cand)

    exact = prefix_ok = empty = mismatch = 0
    missing = 0
    print(f"\n=== diff {gold['stack']}-{gold['scenario']}-n{gold['n']} vs {cand['stack']}-{cand['scenario']}-n{cand['n']} ===\n")
    for qid in sorted(set(gc) | set(cc)):
        g = gc.get(qid)
        c = cc.get(qid)
        if g is None or c is None:
            missing += 1
            print(f"  [{qid}] MISSING gold={g is not None} cand={c is not None}")
            continue
        gtext = (g["content"] or "").strip()
        ctext = (c["content"] or "").strip()
        if ctext == "":
            empty += 1
            status = "EMPTY"
        elif ctext == gtext:
            exact += 1
            status = "EXACT"
        elif ctext[:32] == gtext[:32] and gtext:
            prefix_ok += 1
            status = "PREFIX-OK"
        else:
            mismatch += 1
            status = "MISMATCH"
        det_gold = g["determinism"]
        det_cand = c["determinism"]
        tag = ""
        if det_cand != "unanimous":
            tag += " [cand non-deterministic across slots]"
        if status in ("MISMATCH", "EMPTY"):
            print(f"  [{qid}] {status}{tag}")
            print(f"       gold: {gtext[:120]!r}")
            print(f"       cand: {ctext[:120]!r}")
        else:
            print(f"  [{qid}] {status}{tag}  gold={gtext[:60]!r}")

    total = len(gc | cc)
    print(f"\n  summary: exact={exact}/{total}  prefix={prefix_ok}/{total}  "
          f"empty={empty}/{total}  mismatch={mismatch}/{total}  missing={missing}")
    ok = empty == 0 and mismatch == 0 and missing == 0
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_cap = sub.add_parser("capture")
    p_cap.add_argument("--stack", choices=["pie", "stockvllm"], required=True)
    p_cap.add_argument("--scenario", choices=["s1", "s2"], required=True)
    p_cap.add_argument("--n", type=int, required=True, help="concurrent slots per query")
    p_cap.add_argument("--base-url", required=True)
    p_cap.add_argument("--model", required=True)
    p_cap.add_argument("--seed", type=int, default=42)
    p_cap.add_argument("--max-tokens", type=int, default=96)
    p_cap.add_argument("--timeout", type=float, default=60.0)
    p_cap.add_argument("--bearer", default=None)
    p_cap.add_argument("--unique-session", action="store_true",
                       help="set a unique X-Pie-Session-Id per (query, slot)")
    p_cap.add_argument("--out", required=True)

    p_diff = sub.add_parser("diff")
    p_diff.add_argument("--golden", required=True)
    p_diff.add_argument("--candidate", required=True)

    args = ap.parse_args()
    if args.cmd == "capture":
        cmd_capture(args); return 0
    return cmd_diff(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
