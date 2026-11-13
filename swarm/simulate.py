"""Offline preview of the four configurations on a schedule, before any GPU time.

    python -m swarm.simulate schedules/moto-4x5 --l1-gb 8,32 --l2-gb 0

Replays the schedule through LMCache's chunking rules and reports, per
configuration, where each input token would have come from:

    A  prefix reuse    local L1 per worker
    B  CacheBlend      local L1 per worker
    C  prefix reuse    local L1 + shared Valkey L2
    D  CacheBlend      local L1 + shared Valkey L2  (--fleet-blend: coordinator lookup)

Model:

* A stored prompt becomes chunks of --chunk-size tokens (default 256; sweep
  64,128,256) counted from its first token, each keyed by a prefix-chained hash
  and indexed by its content. Capacity in GiB becomes chunks at
  --kv-bytes-per-token x chunk size.
* Prefix lookup walks the chain from the start, through L1 then L2, and stops
  at the first missing chunk (LMCache's lookup_hit = L1 part + L2 part).
* CacheBlend lookup then probes every token offset past the prefix for a
  stored chunk with the same content, greedily and without overlap. Without a
  coordinator a worker only matches chunks it stored itself (the local
  matcher); with --fleet-blend it matches any chunk resident in L2 or its own L1.
* Every request stores all its full chunks in its worker's L1 (LRU) and, with
  Valkey, in L2 (LRU, 0 = unbounded). Chunks loaded from L2 stay in L1.
* Time: a request takes new_tokens / prefill rate + output / decode rate, then
  the agent waits the schedule's gap and idle time. There is no queueing.

What it leaves out: vLLM's GPU prefix cache, queueing and contention, and the
tokens CacheBlend recomputes (reported separately via --recompute-ratio).
"""

from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
import argparse
import heapq
import json

import numpy as np

from swarm import schedule
from swarm.tokens import CHUNK, Tokenizer, aligned_chunk_hashes, chain, window_hashes

CONFIGS = {
    "A": ("prefix", False),
    "B": ("blend", False),
    "C": ("prefix", True),
    "D": ("blend", True),
}


class LRU:
    def __init__(self, capacity: int):
        self.cap = capacity  # chunks; 0 = unbounded
        self.d: OrderedDict[int, None] = OrderedDict()
        self.evicted = 0
        self.inserted = 0  # chunks newly stored (not already resident)
        self.peak = 0

    def __contains__(self, k: int) -> bool:
        return k in self.d

    def touch(self, k: int) -> None:
        if k in self.d:
            self.d.move_to_end(k)
            return
        self.d[k] = None
        self.inserted += 1
        if self.cap and len(self.d) > self.cap:
            self.d.popitem(last=False)
            self.evicted += 1
        self.peak = max(self.peak, len(self.d))


@dataclass
class Prompts:
    """Per-request token arrays, built from per-message tokens."""

    tok: Tokenizer
    agents: list[schedule.Agent]

    def __post_init__(self) -> None:
        self.placeholder = np.array(self.tok.message("user", schedule.CLEARED_TEXT), dtype=np.int64)
        self.msg_tokens = [
            [np.array(self.tok.message(m["role"], m["content"]), dtype=np.int64) for m in a.messages]
            for a in self.agents
        ]

    def tokens(self, ai: int, k: int) -> np.ndarray:
        t = self.agents[ai].turns[k]
        cleared = set(t.cleared)
        parts = [self.placeholder if i in cleared else self.msg_tokens[ai][i] for i in range(t.msg)]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)


def run(prompts: Prompts, config: str, workers: int, l1_chunks: int, l2_chunks: int,
        fleet_blend: bool, prefill_tps: float, decode_tps: float, chunk: int = CHUNK) -> dict:
    mode, valkey = CONFIGS[config]
    agents = prompts.agents
    l1 = [LRU(l1_chunks) for _ in range(workers)]
    l2 = LRU(l2_chunks) if valkey else None
    local_index: list[dict[int, set[int]]] = [defaultdict(set) for _ in range(workers)]
    fleet_index: dict[int, set[int]] = defaultdict(set)
    owner: dict[int, tuple[int, int]] = {}
    tot: Counter = Counter()
    by_kind: dict[str, Counter] = defaultdict(Counter)
    per_agent_skip = Counter()
    stored_keys: list[set[int]] = [set() for _ in range(workers)]  # what each local matcher indexed
    runs: list[int] = []  # lengths (tokens) of contiguous CacheBlend matches
    # Idle-gap eviction pressure: after each request, the agent's worker and that
    # worker's store/evict counters; at the request after an idle gap, the
    # difference is what the old worker stored and evicted while the agent waited.
    last_seen: dict[int, tuple[int, int, int]] = {}
    gaps: list[dict] = []
    events = [(a.start_s, ai, 0) for ai, a in enumerate(agents)]
    heapq.heapify(events)
    while events:
        now, ai, k = heapq.heappop(events)
        ag = agents[ai]
        turn = ag.turns[k]
        w = turn.worker % workers
        T = prompts.tokens(ai, k)
        L = len(T)
        contents = aligned_chunk_hashes(T, chunk)
        keys, c = [], 0
        for h in contents:
            c = chain(c, h)
            keys.append(c)
        # --- prefix lookup --------------------------------------------------
        p1 = p2 = 0
        for key in keys:
            if key in l1[w]:
                if p2:  # a later L1 chunk after L2 chunks still counts as L2-assisted prefix
                    p2 += 1
                else:
                    p1 += 1
            elif l2 is not None and key in l2:
                p2 += 1
            else:
                break
        pre = (p1 + p2) * chunk
        hit_keys = keys[: p1 + p2]
        # --- CacheBlend lookup ------------------------------------------------
        b1 = b2 = 0
        src: Counter = Counter()
        if mode == "blend" and L - pre >= chunk:
            wh = window_hashes(T, chunk)
            index = fleet_index if fleet_blend else local_index[w]
            i = pre
            whl = wh.tolist()
            n = len(whl)
            run_start = run_end = -1
            while i < n:
                cands = index.get(whl[i])
                found = None
                if cands:
                    for key in cands:
                        if key in l1[w]:
                            found = (key, 1)
                            break
                        if l2 is not None and key in l2:
                            found = (key, 2)
                    if found:
                        key, tier = found
                        if tier == 1:
                            b1 += 1
                        else:
                            b2 += 1
                        hit_keys.append(key)
                        o = owner.get(key)
                        src["own" if o == (ag.dev, ai) else "same_dev" if o and o[0] == ag.dev else "other_dev"] += chunk
                        if i != run_end:
                            if run_end > run_start >= 0:
                                runs.append(run_end - run_start)
                            run_start = i
                        i += chunk
                        run_end = i
                        continue
                i += 1
            if run_end > run_start >= 0:
                runs.append(run_end - run_start)
        # --- accounting -------------------------------------------------------
        r = {"prefix_l1": p1 * chunk, "prefix_l2": p2 * chunk, "blend_l1": b1 * chunk, "blend_l2": b2 * chunk,
             "blend_chunks_loaded": b1 + b2}
        r["new"] = L - sum(r.values())
        r["tokens"] = L
        for kk, v in src.items():
            r["blend_from_" + kk] = v
        tot.update(r)
        tot["requests"] += 1
        kind = "after_move" if turn_moved(ag, k) else "after_idle" if k and ag.turns[k - 1].idle_s else "first" if k == 0 else "steady"
        by_kind[kind].update(r)
        by_kind[kind]["requests"] += 1
        per_agent_skip[ai] += L - r["new"]
        if k and ag.turns[k - 1].idle_s and ai in last_seen:
            w0, ins0, ev0 = last_seen[ai]
            cap_tokens = l1[w0].cap * chunk
            stored = (l1[w0].inserted - ins0) * chunk
            gaps.append({"gap_s": ag.turns[k - 1].idle_s, "moved": turn.moved,
                         "stored_tokens": stored, "evicted_chunks": l1[w0].evicted - ev0,
                         "pressure": (stored / cap_tokens) if cap_tokens else 0.0,
                         "tokens": L, "served": L - r["new"]})
        # --- store --------------------------------------------------------------
        for key in hit_keys:
            l1[w].touch(key)
        for key, h in zip(keys, contents):
            l1[w].touch(key)
            if l2 is not None:
                l2.touch(key)
            local_index[w][h].add(key)
            stored_keys[w].add(key)
            fleet_index[h].add(key)
            owner.setdefault(key, (ag.dev, ai))
        last_seen[ai] = (w, l1[w].inserted, l1[w].evicted)
        # --- next turn -------------------------------------------------------------
        service = r["new"] / prefill_tps + turn.max_tokens / decode_tps
        if k + 1 < len(ag.turns):
            heapq.heappush(events, (now + service + turn.after_s + turn.idle_s, ai, k + 1))
    blend_tok = sum(runs) or 1
    buckets: dict[str, Counter] = {}
    for g in gaps:
        b = "<0.25" if g["pressure"] < 0.25 else "0.25-1" if g["pressure"] < 1 else ">=1"
        c = buckets.setdefault(b, Counter())
        c["requests"] += 1
        c["moved"] += g["moved"]
        c["tokens"] += g["tokens"]
        c["served"] += g["served"]
    return {"config": config, "chunk": chunk, "totals": dict(tot), "by_kind": {k: dict(v) for k, v in by_kind.items()},
            "l1_evicted_chunks": sum(x.evicted for x in l1), "l2_evicted_chunks": l2.evicted if l2 else 0,
            # The local matcher's table holds 2^20 chunks per server lifetime; past
            # that, new content can no longer be blended (LMCache blend_counters.py).
            "max_chunks_indexed_per_worker": max(len(x) for x in stored_keys),
            "blend_runs": len(runs),
            "blend_tokens_in_runs_under_256": round(100 * sum(x for x in runs if x < 256) / blend_tok, 1),
            "blend_tokens_in_runs_under_512": round(100 * sum(x for x in runs if x < 512) / blend_tok, 1),
            "l2_peak_chunks": l2.peak if l2 else 0,
            # Requests right after an idle gap, by how much the agent's old worker
            # stored during the gap relative to its L1 capacity (row 9).
            "idle_gap_pressure": {b: {"requests": c["requests"], "moved": c["moved"],
                                      "served_pct": round(100 * c["served"] / c["tokens"], 1) if c["tokens"] else None}
                                  for b, c in sorted(buckets.items(), key=lambda kv: ["<0.25", "0.25-1", ">=1"].index(kv[0]))},
            "idle_gaps": gaps}


def turn_moved(ag: schedule.Agent, k: int) -> bool:
    return k > 0 and ag.turns[k].moved


def shares(t: dict, recompute_ratio: float) -> dict:
    n = t.get("tokens", 0) or 1
    out = {k: round(100 * t.get(k, 0) / n, 1) for k in ("prefix_l1", "prefix_l2", "blend_l1", "blend_l2", "new")}
    blend = t.get("blend_l1", 0) + t.get("blend_l2", 0)
    out["skipped"] = round(100 * (t.get("prefix_l1", 0) + t.get("prefix_l2", 0) + blend * (1 - recompute_ratio)) / n, 1)
    out["blend_other_agents"] = round(100 * (t.get("blend_from_same_dev", 0) + t.get("blend_from_other_dev", 0)) / n, 1)
    out["blend_other_devs"] = round(100 * t.get("blend_from_other_dev", 0) / n, 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("schedules", type=Path, nargs="+",
                    help="one or more schedule directories (e.g. the -q0/-q0.5/-q1 outputs of a move sweep)")
    ap.add_argument("--configs", default="ABCD")
    ap.add_argument("--l1-gb", default="32", help="L1 per worker, GiB; comma-separated to sweep")
    ap.add_argument("--l2-gb", type=float, default=0, help="Valkey capacity, GiB (0 = unbounded)")
    ap.add_argument("--workers", type=int, default=0, help="0 = from the schedule")
    ap.add_argument("--kv-bytes-per-token", type=int, default=98304,
                    help="default: Qwen3-Coder-30B-A3B bf16 (2 x 48 layers x 4 KV heads x 128 x 2 B)")
    ap.add_argument("--fleet-blend", action="store_true", help="D matches chunks any worker stored (coordinator)")
    ap.add_argument("--prefill-tps", type=float, default=20000.0)
    ap.add_argument("--decode-tps", type=float, default=60.0)
    ap.add_argument("--recompute-ratio", type=float, default=0.15)
    ap.add_argument("--chunk-size", default="256", help="tokens per chunk; comma-separated to sweep (e.g. 64,128,256)")
    ap.add_argument("--tokenizer", default="", help="default: the schedule's")
    ap.add_argument("-o", "--output", type=Path, help="write results JSON here")
    a = ap.parse_args()
    results = []
    tokenizers: dict[str, Tokenizer] = {}
    for sched in a.schedules:
        results += simulate_one(a, sched, tokenizers)
    if a.output:
        a.output.write_text(json.dumps({"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
                                        "results": results}, indent=2, default=str))


def simulate_one(a: argparse.Namespace, sched: Path, tokenizers: dict) -> list[dict]:
    meta, agents = schedule.load(sched)
    name = a.tokenizer or meta.get("tokenizer")
    tok = tokenizers.setdefault(name, Tokenizer(name))
    workers = a.workers or meta["args"].get("workers", 1)
    prompts = Prompts(tok, agents)
    results = []
    q = meta.get("move_prob", meta["args"].get("move_prob"))
    print(f"\n{sched.name}: {meta['repo']}, {len(agents)} agents, {meta['requests']} requests, {workers} workers, "
          f"move probability {q}, {meta.get('idle_gaps', '?')} idle gaps, {meta.get('moves', '?')} moves, "
          f"KV {a.kv_bytes_per_token / 1024:.0f} KiB/token, tokenizer {tok.name}")
    print(f"{'chunk':>5} {'L1/wkr':>7} {'cfg':>3} {'pfx L1':>7} {'pfx L2':>7} {'blnd L1':>7} {'blnd L2':>7} "
          f"{'new':>6} {'skipped':>8} {'blend<-others':>13} {'blend<256':>9} {'idx/wkr':>8} {'L2 peak':>8}")
    for chunk in [int(x) for x in a.chunk_size.split(",")]:
        chunk_bytes = a.kv_bytes_per_token * chunk
        for l1_gb in [float(x) for x in a.l1_gb.split(",")]:
            l1_chunks = int(l1_gb * 2**30 // chunk_bytes)
            l2_chunks = int(a.l2_gb * 2**30 // chunk_bytes)
            for cfg in a.configs:
                res = run(prompts, cfg, workers, l1_chunks, l2_chunks, a.fleet_blend and cfg == "D",
                          a.prefill_tps, a.decode_tps, chunk)
                res.update(schedule=str(sched), move_prob=q, l1_gb=l1_gb, l2_gb=a.l2_gb, chunk_bytes=chunk_bytes,
                           shares=shares(res["totals"], a.recompute_ratio),
                           shares_by_kind={k: shares(v, a.recompute_ratio) for k, v in res["by_kind"].items()})
                s = res["shares"]
                short = f"{res['blend_tokens_in_runs_under_256']:>8}%" if res["blend_runs"] else f"{'-':>9}"
                peak = f"{res['l2_peak_chunks'] * chunk_bytes / 2**30:.0f}G" if res["l2_peak_chunks"] else "-"
                print(f"{chunk:>5} {l1_gb:>7g} {cfg:>3} {s['prefix_l1']:>7} {s['prefix_l2']:>7} {s['blend_l1']:>7} "
                      f"{s['blend_l2']:>7} {s['new']:>6} {s['skipped']:>8} {s['blend_other_agents']:>13} "
                      f"{short} {res['max_chunks_indexed_per_worker']:>8} "
                      f"{peak:>8}")
                results.append(res)
    print("(% of input tokens; skipped = prefix + blend x (1 - recompute ratio); blend<256 = share of CacheBlend "
          "tokens in matches shorter than 256 tokens; idx/wkr = chunks a local matcher indexed, cap 2^20; "
          "L2 peak = Valkey capacity this run needed, GiB)")
    gap_rows = [r for r in results if r["idle_gap_pressure"]]
    if gap_rows:
        print("\nRequests after an idle gap, by what the agent's old worker stored during the gap / its L1 size "
              "(% of their input tokens served from cache; n = requests):")
        for r in gap_rows:
            cells = "  ".join(f"{b}: {v['served_pct']}% (n={v['requests']})" for b, v in r["idle_gap_pressure"].items())
            print(f"  chunk {r['chunk']:>3} L1 {r['l1_gb']:g} {r['config']}  {cells}")
    return results


if __name__ == "__main__":
    main()
