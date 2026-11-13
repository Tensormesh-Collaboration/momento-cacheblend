"""Prometheus scraping for LMCache and vLLM counters, and their deltas.

LMCache's OpenTelemetry exporter turns an instrument such as
`lmcache_mp.lookup_hit_l2` into names like `lmcache_mp_lookup_hit_l2_tokens_total`
(dots to underscores, unit and `_total` suffixes, depending on the version).
`counter()` matches an instrument by stem plus any of those suffixes, the same
rule the NetApp kit's summarizer uses, so a different exporter version still
reads, and a missing counter reads as None rather than zero.
"""

from __future__ import annotations

import urllib.error
import urllib.request

SUFFIXES = ("", "_total", "_tokens", "_tokens_total", "_chunks", "_chunks_total", "_keys", "_keys_total",
            "_requests", "_requests_total", "_1", "_1_total")

# Instruments the summary reads (see lmcache/v1/mp_observability/subscribers/metrics/).
LMCACHE = {
    "requested": "lmcache_mp_lookup_requested",           # chunk-aligned tokens looked up
    "hit": "lmcache_mp_lookup_hit",                       # contiguous prefix hit, L1 + L2
    "hit_l1": "lmcache_mp_lookup_hit_l1",
    "hit_l2": "lmcache_mp_lookup_hit_l2",
    "prompt": "lmcache_mp_lookup_prompt",                 # whole prompt length
    "request_hit": "lmcache_mp_lookup_request_hit",       # prefix served by engine cache or LMCache
    "l1_evicted": "lmcache_mp_l1_evicted",
    "l2_loaded": "lmcache_mp_l2_prefetch_load_completed",
    "l2_stored": "lmcache_mp_l2_store_completed_objects",
    "blend_requested": "lmcache_blend_lookup_requested_tokens",
    "blend_prefix_hit": "lmcache_blend_lookup_prefix_hit_tokens",
    "blend_non_prefix_hit": "lmcache_blend_lookup_non_prefix_hit_tokens",
    "blend_segmented_hit": "lmcache_blend_lookup_segmented_prefix_hit_tokens",
    "blend_stale": "lmcache_blend_lookup_stale_chunks",
}
VLLM = {
    "prompt_tokens": "vllm:prompt_tokens",
    "prefix_cache_queries": "vllm:prefix_cache_queries",
    "prefix_cache_hits": "vllm:prefix_cache_hits",
}


def scrape(url: str, prefix: str) -> dict[str, float] | None:
    """Sum every series whose name starts with prefix, across label sets."""
    try:
        with urllib.request.urlopen(url, timeout=10) as r:  # noqa: S310 - operator-provided URL
            text = r.read().decode(errors="replace")
    except (urllib.error.URLError, OSError, ValueError):
        return None
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#") or not line.startswith(prefix):
            continue
        if "{" in line:
            name, _, rest = line.partition("{")
            value = rest.rpartition("}")[2]
        else:
            name, _, value = line.partition(" ")
        name = name.strip()
        if name.endswith(("_bucket", "_created", "_sum", "_count")):
            continue
        try:
            out[name] = out.get(name, 0.0) + float(value.split()[0])
        except (ValueError, IndexError):
            continue
    return out


def delta(before: dict | None, after: dict | None) -> dict[str, float]:
    if not after:
        return {}
    before = before or {}
    return {k: after[k] - before.get(k, 0.0) for k in after}


def counter(d: dict[str, float], stem: str) -> float | None:
    vals = [v for k, v in d.items() if any(k == stem + s for s in SUFFIXES)]
    return sum(vals) if vals else None


def named(d: dict[str, float], table: dict[str, str]) -> dict[str, float | None]:
    return {k: counter(d, stem) for k, stem in table.items()}
