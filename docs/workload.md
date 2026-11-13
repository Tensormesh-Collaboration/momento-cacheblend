# The swarm workload

Full reasoning, data comparison and open decisions are in the design doc
(linked from the README). This page is the short version for people running it.

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

`python3 -m swarm.simulate`, moto repo; % of input tokens. "Skipped" counts
CacheBlend hits at 85% (15% recompute).

| Schedule | L1 / worker | A | B | C | D |
|---|---|---|---|---|---|
| 4 × 5, long sessions (678 requests) | 8 GiB | 79.4 | 86.0 | 96.4 | 97.1 |
| 4 × 5, long sessions | 32 GiB | 93.7 | 95.2 | 96.4 | 97.1 |
| 8 × 5, sub-agents, 6 turns, 4-file bundle (240 requests) | 8 GiB | 74.5 | 83.9 | 85.4 | 92.3 |
| 8 × 5, sub-agents | 32 GiB | 81.4 | 88.9 | 85.4 | 92.3 |
| 4 × 5, tool-result clearing over 24K, keep 3 | 8 GiB | 88.7 | 90.5 | 93.2 | 94.9 |

What it says, before any GPU time:

- In long sessions with room in L1, prefix caching already skips ~94%:
  CacheBlend adds ~1.5 points, Valkey ~2.7.
- Under L1 pressure Valkey recovers almost everything as prefix hits (C), and
  CacheBlend adds little on top (D − C = 0.7). CacheBlend's gain is larger
  without Valkey (B − A = 6.6), the opposite of the draft blog's Figure 4 claim.
- With short-lived sub-agents both layers matter: each adds roughly 10 points
  alone and D computes about a quarter of A's new tokens (6.5% against 25.5%).

The simulator has no queueing, no vLLM GPU prefix cache, and an idealized
LRU; treat it as a way to pick the legs worth running, not as results.

## Chunk size: 256, 128, 64

`python3 -m swarm.simulate <schedule> --chunk-size 256,128,64`. Same
schedules and assumptions as above; the share of input tokens served by
CacheBlend (non-prefix hits), and what D still computes:

| Schedule, L1 / worker | Non-prefix, B (256 → 128 → 64) | Non-prefix, D | New tokens, D | D's CacheBlend tokens from matches < 256 at chunk 64 |
|---|---|---|---|---|
| Long sessions, 8 GiB | 7.4 → 7.8 → 8.1 | 0.8 → 1.0 → 1.1 | 2.7 → 2.4 → 2.2 | 13.8% |
| Long sessions, 32 GiB | 1.8 → 2.0 → 2.1 | 0.9 → 1.0 → 1.1 | 2.7 → 2.4 → 2.2 | 13.8% |
| Sub-agents, 8 GiB | 12.8 → 12.9 → 13.0 | 8.0 → 8.2 → 8.4 | 6.5 → 6.1 → 5.8 | 1.7% |
| Tool-result clearing, 8 GiB | 2.2 → 2.6 → 2.9 | 1.9 → 2.2 → 2.4 | 4.8 → 4.2 → 3.8 | 12.9% |

Why the gain is small and where it comes from: a stored prompt is cut into
chunks from its first token, so a reused span of L tokens keeps on average
L − C + 1 of them (it loses the partial chunks at its edges). A 2,500-token
file keeps ~90% at 256, ~95% at 128, ~97.5% at 64; a 600-token span keeps
~57%, ~79%, ~90%. Most reusable text here sits in long spans (whole files,
bundles), so halving the chunk adds a few percent to CacheBlend's share; where
spans are shorter and more numerous (cleared histories) it adds up to a third.
At 64, matches shorter than 256 tokens appear (in D, 13-14% of CacheBlend's
tokens in the long-session and clearing schedules, under 2% with sub-agents):
boilerplate and partial file views, the short spans the CacheBlend error model
says reuse least accurately.

What smaller chunks cost (none of it is in the simulator): 2x / 4x more stored
objects and lookups per token, 12 / 6 MiB values instead of 24 MiB through
Valkey and RDMA, and the local matcher's 2^20-chunk table fills 4x sooner
(67M stored tokens per server at 64). The table was never close in these runs
(at most ~9K chunks per worker).

GPU smoke test at each size: `scripts/chunk_smoke.sh --legs "B D"` (one worker,
boots each size, checks that CacheBlend reuses a shifted body). Throughput
needs `run_leg.sh` with `CHUNK_SIZE` set; quality needs `lmcache bench quality`
at each size before 128 or 64 is used for a measured result.
