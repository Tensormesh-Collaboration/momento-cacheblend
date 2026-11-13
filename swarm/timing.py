"""Gap distributions: the time between an agent's reply and its next request.

Specs, as passed to `generate --gap`:

    const:2.0                       always 2 s
    lognormal:median=4,sigma=1.2    a lognormal with that median (s)
    empirical:path.json             resample a JSON list of seconds

`python -m swarm.timing semianalysis <parquet>... -o gaps.json` builds the
empirical list from the SemiAnalysis Claude Code traces
(https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126,
Apache-2.0). Each main-agent request there carries `think_time`: seconds from
the previous response to this request, i.e. tool, test and human time. In the
shard we checked (79 sessions, 8,416 gaps): median 4.2 s, p90 125 s, p95 216 s;
14.6% over 60 s, 4.0% over 5 min, 2.7% over 10 min.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import random
from typing import Callable

Sampler = Callable[[random.Random], float]


def parse(spec: str) -> Sampler:
    kind, _, rest = spec.partition(":")
    if kind == "const":
        v = float(rest)
        return lambda rng: v
    if kind == "lognormal":
        kv = dict(p.split("=") for p in rest.split(","))
        mu, sigma = math.log(float(kv["median"])), float(kv.get("sigma", 1.0))
        cap = float(kv.get("max", "inf"))
        return lambda rng: min(cap, rng.lognormvariate(mu, sigma))
    if kind == "empirical":
        xs = [float(x) for x in json.loads(Path(rest).read_text())]
        if not xs:
            raise ValueError(f"{rest}: empty list")
        return lambda rng: rng.choice(xs)
    raise ValueError(f"unknown gap spec {spec!r} (const:, lognormal:, empirical:)")


def semianalysis(paths: list[Path], cap_s: float) -> dict:
    """Main-agent think times and sub-agent sizes from the SemiAnalysis traces."""
    import pyarrow.parquet as pq  # type: ignore

    think: list[float] = []
    sub_sizes: list[int] = []
    sessions = 0
    for p in paths:
        for sess in pq.read_table(p, columns=["requests"]).column("requests").to_pylist():
            sessions += 1
            for raw in sess:
                q = json.loads(raw) if isinstance(raw, str) else raw
                if q.get("type") == "subagent":
                    sub_sizes.append(len(q.get("requests") or []))
                elif q.get("think_time") is not None:
                    think.append(min(cap_s, float(q["think_time"])))
    return {"sessions": sessions, "think_time_s": think, "subagent_requests": sub_sizes}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("semianalysis", help="extract think-time gaps from the SemiAnalysis parquet")
    s.add_argument("parquet", nargs="+", type=Path)
    s.add_argument("-o", "--output", type=Path, required=True, help="JSON list of gaps (s) for empirical:")
    s.add_argument("--cap-s", type=float, default=3600.0, help="cap each gap (default 1 h)")
    a = ap.parse_args()
    d = semianalysis(a.parquet, a.cap_s)
    a.output.write_text(json.dumps(d["think_time_s"]))
    th = sorted(d["think_time_s"])
    q = lambda p: th[int(p * (len(th) - 1))]  # noqa: E731
    print(f"{d['sessions']} sessions, {len(th)} gaps: p50 {q(.5):.1f}s p90 {q(.9):.1f}s p99 {q(.99):.0f}s; "
          f"{len(d['subagent_requests'])} sub-agent groups -> {a.output}")


if __name__ == "__main__":
    main()
