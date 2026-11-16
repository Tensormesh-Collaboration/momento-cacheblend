# Design and methodology

What the benchmark measures, how, and why it is set up the way it is. The
workload itself (data, request anatomy, simulator results) is in
[workload.md](workload.md); how to run it is in the [README](../README.md).

## The question

A fleet of coding agents re-sends most of its context on every turn. How much
of that prefill does LMCache skip, and how much do two additions buy:

- **CacheBlend**: reuse KV for text that appears again at a different position
  (a file another agent already read, a sub-agent's context bundle), not only
  for an unchanged prefix.
- **A shared Valkey tier** behind each worker's local CPU memory: KV that was
  evicted locally, or stored by another worker, comes back from Valkey instead
  of being recomputed.

## Configurations

| Config | KV reuse | Storage |
|---|---|---|
| A (baseline) | LMCache prefix | Local CPU memory (L1) |
| B | LMCache CacheBlend | L1 |
| C | LMCache prefix | L1 + Valkey |
| D | LMCache CacheBlend | L1 + Valkey |

Everything else is identical across the four legs: model, engine flags,
schedule, L1 size, request timing and the number of tokens generated.

## Workload

Many coding agents work on one repository: N developers, M agents each.
Every agent replays one recorded SWE-agent session from SWE-smith
(teacher-forced: the next turn always sends the recorded history), so every leg
sees exactly the same prompts. Around those sessions the swarm is synthetic and
seeded: a shared harness prompt, per-developer preambles, optional sub-agents
with context bundles, optional clearing of old tool results, think-time gaps,
idle gaps and worker moves. The schedule directory written by
`swarm.generate` is the single input to every leg and to the simulator.
Details: [workload.md](workload.md).

## Methodology

### Serving

- Model: Qwen3-Coder-30B-A3B-Instruct, bf16 (96 KiB of KV per token, so a
  256-token chunk is 24 MiB).
- One worker = one GPU running vLLM and the LMCache server in one container
  (they share GPU memory over CUDA IPC). Workers on a host share nothing but
  the host.
- Versions: LMCache engine image v0.5.8rc1 and cacheblend-plugin v0.5.8rc2
  (rc1 plus one fix; engine and plugin must come from the same release line).

### vLLM's GPU prefix cache is off

All four legs run with vLLM's automatic prefix caching disabled
(`ENABLE_APC=0`), so LMCache's tiers can be measured directly: every reused
token is served by LMCache, from L1 or from Valkey, by prefix or by CacheBlend.

LMCache still does prefix caching with it off: it finds the longest stored
prefix of each prompt and loads that KV into the GPU. What changes is that each
turn reloads its prefix from CPU memory instead of finding it already on the
GPU, which adds load time to TTFT in every leg alike.

Why: on an H200 the GPU prefix cache holds about 680K tokens of this model's KV,
more than the L1 sizes that put LMCache under pressure. With it on, vLLM serves
repeats itself until the GPU cache overflows, so at the scale of a one- or
two-GPU run the legs would measure vLLM rather than LMCache. In a first GPU run
with it on, LMCache served 0% of prompt tokens; with it off, 74% (leg A). At
production scale a swarm's combined context (40 agents with 30K-token contexts
is 1.2M tokens) outgrows the GPU cache anyway, which then only holds the
hottest agents; turning it off removes that layer from every leg equally.
Report this setting with the results.

### Engine flags

CacheBlend (plugin v0.5.8rc1) needs chunked prefill off and either decode-only
CUDA graphs (`--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'`,
the kit's default) or `--enforce-eager`. With `FLAG_MATCHED=1` (default) the
prefix legs A and C run with the same flags, so hit rates and latencies compare
like for like. `FLAG_MATCHED=0` gives the prefix legs vLLM's production
defaults (full CUDA graphs, chunked prefill): use it only for a goodput
comparison, and label it. vLLM warns that this model does not officially
support disabling chunked prefill; no failures were seen in testing.

### Warm start

Every leg starts from empty caches: fresh workers (empty L1) and this kit's
Valkey keys deleted (only keys under `VALKEY_KEY_PREFIX@`, never the whole
database). It then warms them before measuring, by default: one extra
developer's agents (the warm-up cohort, `swarm.generate --warmup-devs 1`) are
replayed first, back to back and untimed, across all workers. They work on the
same repository with tasks no measured agent uses, so they leave behind what a
fleet that has been running would: the shared harness prompt and the repo's
commonly read files, in L1 and in Valkey. The leg is measured from when they
finish: counter deltas, wall time and the measured agents' start offsets all
count from there, and their requests are left out of every result.

Without it, every leg opens with the misses no cache can avoid (each agent's
first turn, each file the first time anyone reads it). Those shrink every
leg's numbers and favour whichever configuration fills fastest, while a real
fleet runs continuously. The measured agents are byte-for-byte the same with
or without the cohort, so warm and cold runs compare directly:
`WARM_START=0` (or `scripts/sweep.sh --start cold`) measures from empty caches.
In the simulator the warm-up moves results by under 1 point on the main
schedules and by up to 2 points on the small first-run schedule.

### Replies

Each turn generates the recorded reply length (capped at 512 tokens) with
`ignore_eos` and temperature 0, and the output is discarded: the next turn
sends the recorded history. Decode load is therefore identical in every leg,
and the harness measures prefill reuse and latency, not answer quality.

### L1 sizing

L1 per worker is the capacity axis: it is set below the swarm's working set,
so A and B evict and recompute, and C and D spill to Valkey. Valkey is sized
above the working set (the simulator's `L2 peak`: 59-77 GiB for the schedules
in [workload.md](workload.md)), and the summary reports Valkey's own evictions
during each leg, which must be 0.

How LMCache's L1 behaves, measured on GPUs, decides what an L1 size means:

- It evicts in a background loop, once a second, and only once L1 is fuller
  than `EVICTION_WATERMARK`; it then frees `EVICTION_RATIO` (0.2) of the
  resident chunks, least recently used first. So L1 caches about watermark x
  `L1_SIZE_GB`, not all of it.
- A request's chunks are allocated in one batch. If they do not fit in what is
  free, none of them is stored (in L1 or Valkey) or loaded, and nothing is
  evicted to make room. So `(1 - EVICTION_WATERMARK) x L1_SIZE_GB` must hold
  the largest prompt; at 64K tokens that is 6 GiB. `validate_config` refuses
  settings that break this, against `MAX_REQUEST_TOKENS` (default
  `MAX_MODEL_LEN`).

The Momento profile uses 16 GiB at watermark 0.5: about 6.4-8 GiB cached and
8 GiB always free. The simulator models both behaviours (`--l1-gb` is
`L1_SIZE_GB`, `--eviction-watermark` the watermark) and reports requests that
would store nothing.

### Routing, idle gaps and worker moves

Each agent stays on one worker. After an idle gap it moves to another worker
with probability 0.5 (an agent resumed elsewhere after a restart or
rebalance); moves are fixed in the schedule, so every leg sees the same ones.
An idle gap only costs something if the agent's KV is evicted while it waits,
and LMCache evicts the least recently used KV, so the rest of the swarm keeps
running during each gap and results after a gap are grouped by eviction
pressure: how much the agent's old worker stored into L1 while it waited,
relative to what L1 caches. An optional sweep over move probabilities 0, 0.5
and 1 is built in.

### Chunk size

256 tokens, CacheBlend's validated granularity. 128 and 64 are available for a
sweep (`CHUNK_SIZE`); the simulator expects a few percent more CacheBlend reuse
at 64, and their accuracy needs `lmcache bench quality` before they are
reported.

## Measurement

- **Reused tokens**: LMCache's own counters, read from each worker's metrics
  endpoint before and after the leg: CacheBlend's prefix and non-prefix hit
  tokens in B and D, LMCache's request-level prefix hit in A and C.
- **Skipped tokens** = prefix hits + non-prefix hits x (1 - recompute ratio):
  CacheBlend recomputes part of what it loads (15% by default).
- **Tier split**: LMCache's L1 and L2 lookup hits (prefix lookups). DRAM
  against NVMe inside Valkey is not visible to LMCache; it needs the
  large-object module's own stats.
- **Latency**: TTFT per request, overall and for requests after an idle gap or
  a move; goodput is requests under a TTFT target per second.
- **Failed KV loads**: with `KV_LOAD_FAILURE_POLICY=recompute` (every leg), a
  KV load that fails is recomputed instead of failing the request; the summary
  counts them from the worker logs (`failed_kv_loads`) and warns about LMCache
  L1 read-race and allocation errors (Tensormesh LM-1157). They cost hits, so a
  leg with many of them understates that configuration.
- **Per run**: the configuration (`config.env`), warm or cold start, Valkey's
  evictions, and an idle-gap table from LMCache counters the replay samples
  every 5 s.

## Decisions

| # | Decision | Setting |
|---|---|---|
| 1 | Realism knobs in the headline run | Sub-agent mode, idle gaps and moves on |
| 2 | Software | LMCache and cacheblend-plugin v0.5.8rc1. Valkey over RDMA (EFA) is the target, TCP the tested default |
| 3 | Fair baselines | Hit rates with matched engine flags; goodput on vLLM defaults, labelled |
| 4 | What counts as a hit | LMCache's counters, with vLLM's GPU prefix cache off; skipped tokens as above |
| 5 | Unit and quality | A task is one agent session; accuracy is out of scope for this harness |
| 6 | Tier split inside Valkey | From the large-object module's stats, per run |
| 7 | Capacity axis | L1 per worker, below the working set; Valkey above it, evictions checked |
| 8 | Warm or cold start | Warm: one untimed warm-up developer first (default); cold as a switch |
| 9 | Idle gaps | The swarm keeps running during gaps; results grouped by eviction pressure |
| 10 | Routing | Sticky, move after an idle gap with probability 0.5; optional sweep 0 / 0.5 / 1 |
| 11 | Chunk alignment | Natural offsets (files lose their partial edge chunks) |
| 12 | Harness prompt | One shared 8K-token system prompt (a knob) |
| 13 | Tokenizer | Schedules are built with the serving model's tokenizer and chat template |
| 14 | Output length | Recorded reply length, capped at 512, `ignore_eos` |
| 15 | Data licences | SWE-smith trajectories (MIT) and SemiAnalysis timing (Apache-2.0) only |
| 16 | Who runs it | Momento, on their AWS hosts, with a Tensormesh key |

## Limitations

- Teacher-forced replay: agents never act on the model's own output, so task
  success is not measured.
- Greedy outputs are not byte-identical between a cache hit and a full prefill
  even in plain vLLM on this MoE model, so accuracy has to come from a quality
  benchmark, not from comparing outputs across legs.
- CacheBlend's index can point at chunks already evicted from L1; those
  lookups find nothing ("stale"), which lowers B's measured non-prefix reuse
  below the simulator's.
- See the README's status table for what has and has not been run on GPUs.
