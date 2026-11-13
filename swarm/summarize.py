"""Compare replay runs: latency, goodput and where reused tokens came from.

    python -m swarm.summarize runs/A runs/B runs/C runs/D --ttft-target 2.0

Token accounting (decision 4 in the design doc):

* reused tokens = CacheBlend's prefix + non-prefix hit tokens when the blend
  counters are present, else LMCache's request-level hit
  (`lmcache_mp.lookup_request_hit`: prefix served by the engine cache or LMCache).
* skipped tokens = prefix hits + non-prefix hits x (1 - recompute ratio),
  since CacheBlend recomputes part of what it loads.
* tier split = LMCache's lookup_hit_l1 / lookup_hit_l2 (prefix lookups).
  DRAM against NVMe inside Valkey is not visible here: it needs the
  large-object module's own stats.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json


def pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def summarize(run: Path, ttft_target: float, recompute_ratio: float) -> dict:
    reqs = [json.loads(line) for line in open(run / "requests.jsonl")]
    meta = json.loads((run / "run.json").read_text())
    m = json.loads((run / "metrics.json").read_text())
    ok = [r for r in reqs if r["ok"]]
    ttft = [r["ttft_s"] for r in ok]
    wall = meta["ended"] - meta["started"]

    def sub(pred) -> list[float]:
        return [r["ttft_s"] for r in ok if pred(r)]

    def total(section: str, key: str) -> float | None:
        vals = [w[section].get(key) for w in m["workers"]]
        vals = [v for v in vals if v is not None]
        return sum(vals) if vals else None

    prompt = sum(r.get("prompt_tokens", 0) for r in ok)
    bp, bn = total("lmcache", "blend_prefix_hit"), total("lmcache", "blend_non_prefix_hit")
    if bp is not None or bn is not None:
        prefix_hit, non_prefix = bp or 0.0, bn or 0.0
    else:
        prefix_hit, non_prefix = total("lmcache", "request_hit") or 0.0, 0.0
    h1, h2 = total("lmcache", "hit_l1"), total("lmcache", "hit_l2")
    agents = len({r["agent"] for r in reqs})
    skipped = prefix_hit + non_prefix * (1 - recompute_ratio)
    return {
        "run": run.name,
        "requests": len(reqs),
        "failed": len(reqs) - len(ok),
        "wall_s": round(wall),
        "ttft_p50": pct(ttft, 50), "ttft_p99": pct(ttft, 99),
        "ttft_p50_after_idle": pct(sub(lambda r: r["after_idle"]), 50),
        "ttft_p50_after_move": pct(sub(lambda r: r["moved"]), 50),
        "goodput_rps": round(sum(1 for x in ttft if x <= ttft_target) / wall, 3) if wall > 0 else None,
        "prompt_tokens": prompt,
        "reused_pct": round(100 * (prefix_hit + non_prefix) / prompt, 1) if prompt else None,
        "non_prefix_pct": round(100 * non_prefix / prompt, 1) if prompt else None,
        "l2_share_of_prefix_hits_pct": round(100 * h2 / (h1 + h2), 1) if h1 is not None and h2 is not None and (h1 + h2) else None,
        "vllm_gpu_prefix_hit_pct": (round(100 * total("vllm", "prefix_cache_hits") / total("vllm", "prefix_cache_queries"), 1)
                                    if total("vllm", "prefix_cache_queries") else None),
        "skipped_tokens_per_task": round(skipped / agents) if agents else None,
        "lmcache_reachable": all(w["lmcache_reachable"] for w in m["workers"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--ttft-target", type=float, default=2.0, help="seconds; goodput counts requests under it")
    ap.add_argument("--recompute-ratio", type=float, default=0.15)
    ap.add_argument("-o", "--output", type=Path)
    a = ap.parse_args()
    rows = [summarize(r, a.ttft_target, a.recompute_ratio) for r in a.runs]
    cols = ["run", "requests", "failed", "ttft_p50", "ttft_p99", "ttft_p50_after_idle", "ttft_p50_after_move",
            "goodput_rps", "reused_pct", "non_prefix_pct", "l2_share_of_prefix_hits_pct", "skipped_tokens_per_task"]
    fmt = lambda v: "-" if v is None else (f"{v:.3f}" if isinstance(v, float) and v < 10 else str(v))  # noqa: E731
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    if any(not r["lmcache_reachable"] for r in rows):
        print("\nWARN: LMCache /metrics unreachable for some workers: token columns are incomplete.")
    if a.output:
        a.output.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
