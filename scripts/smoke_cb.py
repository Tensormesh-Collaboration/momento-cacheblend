#!/usr/bin/env python3
"""CacheBlend smoke test for one worker: does blending actually fire?

    scripts/smoke_cb.py --port 8000 --metrics http://127.0.0.1:8080/metrics

Sends the same ~1K-token body twice, behind two different 256-token preambles,
so on the second request the body sits behind a prefix the cache has never
seen. Prefix reuse cannot serve it; CacheBlend can. PASS when LMCache's
non-prefix hit counter (lmcache_blend.lookup_non_prefix_hit_tokens) grows on
the second request. Same idea as cacheblend-plugin's model_cards/smoke_test.sh,
but standard library only: prompts are built from token ids that the engine's
own /tokenize returns, so no tokenizer is needed on this host.

Run it first on a model without a plugin model card (Qwen3-Coder-30B-A3B), and
again for each CB_CHECK_LAYER you consider.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from swarm import metrics  # noqa: E402


def call(url: str, body: dict | None = None, timeout: float = 600) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - operator-provided URL
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--metrics", default="http://127.0.0.1:8080/metrics", help="this worker's LMCache /metrics")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--settle", type=float, default=3.0, help="seconds for the first request's store to land")
    a = ap.parse_args()
    base = f"http://{a.host}:{a.port}"
    model = call(f"{base}/v1/models")["data"][0]["id"]

    def ids(text: str) -> list[int]:
        return call(f"{base}/tokenize", {"model": model, "prompt": text, "add_special_tokens": False})["tokens"]

    def preamble(seed: str) -> list[int]:
        t, fill = ids(seed), ids(" filler") or [0]
        while len(t) < a.chunk:
            t += fill
        return t[: a.chunk]

    words = "apple banana cherry orange grape tiger lion elephant zebra panda".split()
    # Unique per run: with a shared L2, another worker's smoke test has already stored a fixed body,
    # and this worker's second request would be a prefix hit from L2 instead of a blend.
    run = secrets.token_hex(3)
    body = ids(" ".join(f"{words[i % len(words)]}{run}{i:04d}" for i in range(220)))
    suffix = ids("\nQ: name one word from the list above.\nA:")
    prompts = {
        "warmup": preamble("Alpha bravo charlie delta echo") + body + suffix,
        "shifted": preamble("Sunrise sunset river ocean meadow") + body + suffix,
    }
    before = metrics.scrape(a.metrics, "lmcache")
    if before is None:
        print(f"FAIL  LMCache /metrics unreachable at {a.metrics}")
        return 1
    snap = {}
    for tag, p in prompts.items():
        t0 = time.perf_counter()
        call(f"{base}/v1/completions", {"model": model, "prompt": p, "max_tokens": 8, "temperature": 0})
        print(f"  {tag:<8} {len(p)} tokens, {time.perf_counter() - t0:.2f}s")
        time.sleep(a.settle)
        snap[tag] = metrics.scrape(a.metrics, "lmcache")
    d = metrics.named(metrics.delta(snap["warmup"], snap["shifted"]), metrics.LMCACHE)
    npx, req = d.get("blend_non_prefix_hit"), d.get("blend_requested")
    print(f"  second request: requested {req}, prefix hit {d.get('blend_prefix_hit')}, non-prefix hit {npx}, "
          f"stale chunks {d.get('blend_stale')}")
    if npx is None:
        print("FAIL  no lmcache_blend_* counters: is this worker running CACHE_MODE=blend?")
        return 1
    if npx <= 0:
        print("FAIL  CacheBlend did not reuse the shifted body (non-prefix hits = 0). "
              "Check the vLLM log for 'CB:' lines and degraded scatters.")
        return 1
    print(f"PASS  CacheBlend reused {npx:.0f} tokens at a shifted position")
    return 0


if __name__ == "__main__":
    sys.exit(main())
