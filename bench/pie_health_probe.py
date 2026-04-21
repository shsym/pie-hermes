"""Sequential 3-request diagnostic for a pie openai-compat-v2 endpoint.

Reproduces the minimal failure pattern from task #22 round 2b where:
  - Request 1 (simple prompt): succeeded
  - Request 2 (same prompt): hung with hyper::Error(IncompleteMessage)

This script runs three requests serially with timings and content,
exiting non-zero if any request errors. Use it as a smoke between pie
deployment and the heavier driver.

Usage: python scenarios/pie_health_probe.py [base_url]
  default base_url: http://localhost:18095/v1
"""
import json
import sys
import time
import urllib.request
import urllib.error

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:18095/v1"


def req(label, body, timeout=30):
    t0 = time.monotonic()
    data = json.dumps(body).encode()
    url = BASE + "/chat/completions"
    r = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(r, timeout=timeout)
        elapsed = time.monotonic() - t0
        payload = json.loads(resp.read())
        choice = payload["choices"][0]["message"]
        content = (choice.get("content") or "")[:120]
        tc = len(choice.get("tool_calls") or [])
        finish = payload["choices"][0].get("finish_reason")
        usage = payload.get("usage") or {}
        print(f"  {label:<22} OK  {elapsed:5.1f}s  finish={finish} tool_calls={tc} "
              f"prompt_tok={usage.get('prompt_tokens')} "
              f"content={content!r}")
        return True
    except Exception as e:
        elapsed = time.monotonic() - t0
        print(f"  {label:<22} FAIL {elapsed:5.1f}s  {type(e).__name__}: {str(e)[:200]}")
        return False


SMALL_NO_TOOLS = {
    "model": "Qwen/Qwen2.5-72B-Instruct-AWQ",
    "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
    "max_tokens": 10,
    "temperature": 0.0,
    "stream": False,
}

SMALL_NO_TOOLS_B = {**SMALL_NO_TOOLS,
                   "messages": [{"role": "user", "content": "Reply with exactly: HI"}]}

WITH_TOOLS = {
    "model": "Qwen/Qwen2.5-72B-Instruct-AWQ",
    "messages": [{"role": "user", "content": "What is 2+2? Answer in one number."}],
    "max_tokens": 15,
    "temperature": 0.0,
    "stream": False,
    "tools": [{
        "type": "function",
        "function": {
            "name": "calc",
            "description": "Arithmetic calculator",
            "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}, "required": ["expr"]},
        },
    }],
    "tool_choice": "auto",
}


def main():
    print(f"Probing {BASE}")
    ok1 = req("req1 no-tools", SMALL_NO_TOOLS)
    ok2 = req("req2 no-tools", SMALL_NO_TOOLS_B)
    ok3 = req("req3 w/tools",  WITH_TOOLS)
    all_ok = ok1 and ok2 and ok3
    print(f"\nresult: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
