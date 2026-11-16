# The swarm workload

## What it reproduces

Many coding agents work on one repository. Each agent's own turns are prefix
hits: every request repeats the previous one and appends a tool call and its
result. Reuse beyond that comes from two places:

- **Files several agents read**, at different positions behind different
  histories. Only CacheBlend can reuse those: it probes every token offset of
  a new prompt for 256-token chunks another prompt stored. A file keeps all but
  its partial edge chunks; a span under 511 tokens is not guaranteed a chunk.
- **An agent's own history after it was lost**: evicted from local memory
  during an idle gap or under capacity pressure, or left behind on another
  worker. A shared Valkey tier returns it as prefix hits.

Sub-agents (a new prompt over files the parent already read) and context
editing (old tool results cleared, shifting everything after them) break the
prefix on purpose; both are knobs.

## Data

- **Content and file access**: [SWE-smith trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories)
  (MIT), SWE-agent runs by Claude 3.7 Sonnet. SWE-smith runs every task in a
  repo against one environment, so a file reads the same for every agent except
  where that task's injected bug changed it. That is what makes "N developers
  on one codebase" cheap: a developer is a group of one repo's tasks.
- **Timing** (optional): [SemiAnalysis Claude Code traces](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126)
  (Apache-2.0), `think_time` between requests. In one shard: median 4.2 s,
  p90 125 s, 14.6% over a minute, 2.7% over ten minutes.

## Request anatomy

```
system:    shared harness prompt (--harness-tokens, default 8K)
user:      developer preamble (--dev-preamble-tokens) + the agent's task (PR text)
           [+ "Context from the lead agent": the developer's top files, sub-agent mode]
assistant: recorded turn 1 (tool call rendered as text)
user:      OBSERVATION: recorded tool result (file views as `cat -n` output)
...        turn k sends everything up to assistant turn k
```

Replies are generated for the recorded length (`ignore_eos`) and discarded:
every leg sees identical prompts and the same decode load.

## Offline preview (Qwen3-Coder tokenizer, unbounded Valkey, 2 workers)

`python -m swarm.simulate <schedule> --l1-gb <L1_SIZE_GB> --eviction-watermark <W>`,
moto repo; % of input tokens skipped. "Skipped" counts CacheBlend hits at 85%
(15% recompute). L1 is modelled the way LMCache runs it (see "L1 sizing" in
[design.md](design.md)): it evicts once fuller than the watermark, so it caches
about watermark x its size, and a prompt that does not fit in what is free is
not stored at all. Every row below keeps the largest prompt of its schedule
inside the free space (`no-store` = 0).

| Schedule | L1 per worker, watermark | A | B | C | D |
|---|---|---|---|---|---|
| 4 x 5, long sessions (678 requests) | 16 GiB, 0.5 | 75.5 | 84.5 | 96.4 | 96.9 |
| 4 x 5, long sessions | 32 GiB, 0.8 | 92.6 | 94.3 | 96.4 | 96.9 |
| **8 x 5, sub-agents, 6 turns, 4-file bundle (240 requests)** | **16 GiB, 0.5** (the profile's default) | **74.6** | **82.3** | **85.4** | **90.9** |
| 8 x 5, sub-agents | 30 GiB, 0.8 | 80.9 | 88.2 | 85.4 | 90.9 |
| 4 x 5, tool-result clearing over 24K, keep 3 | 12 GiB, 0.5 | 85.7 | 87.7 | 93.2 | 94.6 |

What it says, before the measured runs:

- In long sessions with room in L1, prefix caching already skips ~93%:
  CacheBlend adds ~1.7 points, Valkey ~3.8.
- Under L1 pressure Valkey recovers almost everything as prefix hits (C), and
  CacheBlend adds little on top (D - C = 0.5). CacheBlend's gain is much larger
  without Valkey (B - A = 9.0).
- With short-lived sub-agents both layers matter: each adds 8-11 points alone,
  and D computes about a third of A's new tokens (8.2% against 25.4%).
- Valkey needs 59-77 GiB for these schedules (the simulator's `L2 peak`).

How far to trust it: on a first GPU run (one H200 worker, a 72-request 2 x 3
schedule, L1 6 GiB at watermark 0.5, vLLM's GPU prefix cache off), the measured
legs came within 3.5 points of the simulator:

| | A | B | C | D |
|---|---|---|---|---|
| Measured, % of prompt tokens reused | 73.8 | 78.5 | 92.1 | 93.1 |
| Simulator, same schedule and L1 | 70.3 | 78.6 | 92.1 | 93.0 |

The simulator has no queueing and no timing effects on hit rates; use it to
pick L1 sizes and legs, and the GPU runs for results.

## Chunk size: 256, 128, 64

`python -m swarm.simulate <schedule> --chunk-size 256,128,64`. Same
schedules and assumptions as above; the share of input tokens served by
CacheBlend (non-prefix hits), and what D still computes:

| Schedule, L1 per worker, watermark | Non-prefix, B (256 -> 128 -> 64) | Non-prefix, D | New tokens, D | D's CacheBlend tokens from matches < 256 at chunk 64 |
|---|---|---|---|---|
| Long sessions, 16 GiB, 0.5 | 7.9 -> 8.4 -> 8.7 | 0.6 -> 0.7 -> 0.9 | 3.0 -> 2.7 -> 2.4 | 16.2% |
| Long sessions, 32 GiB, 0.8 | 2.3 -> 2.4 -> 2.6 | 0.6 -> 0.7 -> 0.8 | 3.0 -> 2.7 -> 2.4 | 16.2% |
| Sub-agents, 16 GiB, 0.5 | 10.9 -> 10.4 -> 11.4 | 6.4 -> 6.6 -> 6.7 | 8.2 -> 7.7 -> 7.5 | 1.9% |
| Tool-result clearing, 12 GiB, 0.5 | 2.6 -> 3.1 -> 3.3 | 1.6 -> 1.8 -> 2.1 | 5.2 -> 4.6 -> 4.2 | 14.2% |

Why the gain is small and where it comes from: a stored prompt is cut into
chunks from its first token, so a reused span of L tokens keeps on average
L − C + 1 of them (it loses the partial chunks at its edges). A 2,500-token
file keeps ~90% at 256, ~95% at 128, ~97.5% at 64; a 600-token span keeps
~57%, ~79%, ~90%. Most reusable text here sits in long spans (whole files,
bundles), so halving the chunk adds a few percent to CacheBlend's share; where
spans are shorter and more numerous (cleared histories) it adds up to a third.
At 64, matches shorter than 256 tokens appear (in D, 14-16% of CacheBlend's
tokens in the long-session and clearing schedules, under 2% with sub-agents):
boilerplate and partial file views, the short spans the CacheBlend error model
says reuse least accurately.

What smaller chunks cost (none of it is in the simulator): 2x / 4x more stored
objects and lookups per token, 12 / 6 MiB values instead of 24 MiB through
Valkey and RDMA, and the local matcher's 2^20-chunk table fills 4x sooner
(67M stored tokens per server at 64). The table was never close in these runs
(at most ~8.6K chunks per worker at 256).

GPU smoke test at each size: `scripts/chunk_smoke.sh --legs "B D"` (one worker,
boots each size, checks that CacheBlend reuses a shifted body). Throughput
needs `run_leg.sh` with `CHUNK_SIZE` set; quality needs `lmcache bench quality`
at each size before 128 or 64 is used for a measured result.

## Idle gaps and worker moves

An idle gap only costs something if the agent's KV is evicted while it waits,
and LMCache evicts the least recently used KV, not the oldest. So the simulator
and the summary group every request after an idle gap by eviction pressure:
tokens the agent's old worker stored into L1 during the gap, divided by its L1
size. Below 0.25 the agent's KV is very likely still there; at 1 or more, its old
worker turned over all of L1.

Worker moves draw from their own random stream, so a sweep
(`--move-prob 0,0.5,1`) changes nothing but which agents move. On the
sub-agent schedule (8 x 5, 5-minute gaps) at the default L1 (16 GiB, 0.5), 20
of 23 gaps had pressure of 1 or more: the old worker had already lost the
agent's KV, so moving cost nothing extra and A to D barely changed with the
move probability. Moves only matter when L1 is large enough to survive the gap.
At 128 GiB per worker (about 64 GiB cached), all gaps fell below pressure 1,
and moving every agent after its gap (q = 1 against q = 0) cut A from 85.2% to
81.4% of tokens skipped; its post-gap requests fell from 87-97% served to
39-68%. C stayed at 85.4% because those agents found their KV in Valkey. That
is the regime where the optional move sweep shows Valkey's cross-worker value.
