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

Also per run: the sweep coordinates (L1 per worker, chunk size, worker-move
probability), Valkey's own evictions during the leg (must be 0, or Valkey and
not L1 limited the run), and, from metrics_timeseries.jsonl, the requests that
follow an idle gap grouped by eviction pressure: how many tokens the agent's
old worker stored into L1 while it waited, as a fraction of that L1.
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


BUCKETS = ("<0.25", "0.25-1", ">=1")


def bucket(pressure: float) -> str:
    return BUCKETS[0] if pressure < 0.25 else BUCKETS[1] if pressure < 1 else BUCKETS[2]


def read_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def gap_pressure(run: Path, reqs: list[dict], l1_tokens: float, chunk: int) -> list[dict]:
    """One row per request after an idle gap: what its old worker stored during the gap."""
    ts_path = run / "metrics_timeseries.jsonl"
    if not ts_path.exists() or l1_tokens <= 0:
        return []
    series: dict[int, list[tuple[float, float]]] = {}
    for line in open(ts_path):
        d = json.loads(line)
        if d.get("l1_write") is not None:
            series.setdefault(d["worker"], []).append((d["t_s"], d["l1_write"]))
    for v in series.values():
        v.sort()

    def at(worker: int, t: float) -> float | None:
        pts = series.get(worker)
        if not pts:
            return None
        before = [v for ts, v in pts if ts <= t]
        return before[-1] if before else pts[0][1]

    by_turn = {(r["agent"], r["turn"]): r for r in reqs}
    out = []
    for r in reqs:
        if not r.get("after_idle") or not r["ok"]:
            continue
        prev = by_turn.get((r["agent"], r["turn"] - 1))
        if prev is None or "t_end_s" not in prev:
            continue
        w0 = prev["worker"]
        a, b = at(w0, prev["t_end_s"]), at(w0, r["t_submit_s"])
        if a is None or b is None:
            continue
        stored = (b - a) * chunk  # l1_write counts chunks (one key per chunk at TP=1)
        out.append({"agent": r["agent"], "turn": r["turn"], "moved": r["moved"], "ttft_s": r["ttft_s"],
                    "prompt_tokens": r.get("prompt_tokens", 0), "cached_tokens": r.get("cached_tokens"),
                    "gap_s": round(r["t_submit_s"] - prev["t_end_s"], 1), "stored_tokens": stored,
                    "pressure": stored / l1_tokens})
    return out


def summarize(run: Path, ttft_target: float, recompute_ratio: float, kv_bytes_per_token: int = 98304) -> dict:
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
    env = read_env(run / "config.env")
    l1_gb = float(env["L1_SIZE_GB"]) if env.get("L1_SIZE_GB") else None
    chunk = int(env.get("CHUNK_SIZE") or 256)
    sched_meta = meta.get("schedule_meta", {})
    move_prob = sched_meta.get("move_prob", sched_meta.get("args", {}).get("move_prob"))
    vb, va = run / "valkey_before.json", run / "valkey_after.json"
    valkey_evicted = (json.loads(va.read_text())["evicted_keys"] - json.loads(vb.read_text())["evicted_keys"]
                      if vb.exists() and va.exists() else None)
    # L1 evicts above the watermark, so it holds about watermark x L1_SIZE_GB.
    watermark = float(env.get("EVICTION_WATERMARK") or 0.8)
    gaps = gap_pressure(run, reqs, watermark * l1_gb * 2**30 / kv_bytes_per_token if l1_gb else 0, chunk)
    by_bucket = {}
    for b in BUCKETS:
        rows = [g for g in gaps if bucket(g["pressure"]) == b]
        if rows:
            # vLLM's cached_tokens counts tokens from its GPU cache and from LMCache
            served = [g for g in rows if g["cached_tokens"] is not None and g["prompt_tokens"]]
            by_bucket[b] = {"requests": len(rows), "moved": sum(g["moved"] for g in rows),
                            "ttft_p50": pct([g["ttft_s"] for g in rows], 50),
                            "served_pct": (round(100 * sum(g["cached_tokens"] for g in served)
                                                 / sum(g["prompt_tokens"] for g in served), 1) if served else None)}
    return {
        "run": run.name,
        "leg": env.get("CACHE_MODE", "?") + "+" + env.get("L2_BACKEND", "?"),
        "l1_gb": l1_gb,
        "chunk": chunk,
        "move_prob": move_prob,
        "valkey_evicted_keys": valkey_evicted,
        "idle_gap_pressure": by_bucket,
        "idle_gaps": gaps,
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
    ap.add_argument("--kv-bytes-per-token", type=int, default=98304, help="default: Qwen3-Coder-30B-A3B bf16")
    ap.add_argument("-o", "--output", type=Path)
    a = ap.parse_args()
    rows = [summarize(r, a.ttft_target, a.recompute_ratio, a.kv_bytes_per_token) for r in a.runs]
    cols = ["run", "leg", "l1_gb", "chunk", "move_prob", "requests", "failed", "ttft_p50", "ttft_p99", "ttft_p50_after_idle", "ttft_p50_after_move",
            "goodput_rps", "reused_pct", "non_prefix_pct", "l2_share_of_prefix_hits_pct", "skipped_tokens_per_task",
            "valkey_evicted_keys"]
    fmt = lambda v: "-" if v is None else (f"{v:.3f}" if isinstance(v, float) and v < 10 else str(v))  # noqa: E731
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    gap_rows = [r for r in rows if r["idle_gap_pressure"]]
    if gap_rows:
        print("\nRequests after an idle gap, by what the agent's old worker stored in L1 during the gap / L1 size:")
        print("| run | " + " | ".join(f"{b}: n, moved, % served, TTFT p50" for b in BUCKETS) + " |")
        print("|" + "---|" * (len(BUCKETS) + 1))
        for r in gap_rows:
            cells = []
            for b in BUCKETS:
                v = r["idle_gap_pressure"].get(b)
                cells.append(f"{v['requests']}, {v['moved']}, {fmt(v['served_pct'])}, {fmt(v['ttft_p50'])}" if v else "-")
            print(f"| {r['run']} | " + " | ".join(cells) + " |")
    if any((r["valkey_evicted_keys"] or 0) > 0 for r in rows):
        print("\nWARN: Valkey evicted keys during some legs: its capacity, not L1, limited those runs.")
    if any(not r["lmcache_reachable"] for r in rows):
        print("\nWARN: LMCache /metrics unreachable for some workers: token columns are incomplete.")
    if a.output:
        a.output.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
