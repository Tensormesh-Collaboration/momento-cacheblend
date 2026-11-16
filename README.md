# CacheBlend + Valkey: agent-swarm benchmark (Momento x Tensormesh)

Measure how much prefill a fleet of coding agents skips when LMCache reuses KV
by prefix or by CacheBlend, from local memory or from a shared Valkey tier.
The workload is a synthetic swarm built from real SWE-smith agent sessions:
N developers, M agents each, one shared codebase, with idle gaps and agents
moving between inference workers.

| Config | Reuse | Storage |
|---|---|---|
| A (baseline) | LMCache prefix | Local CPU offload (L1) |
| B | LMCache CacheBlend | Local CPU offload (L1) |
| C | LMCache prefix | L1 + Valkey |
| D | LMCache CacheBlend | L1 + Valkey |

> Private repository for the Momento engagement. The serving plumbing (setup,
> plugin staging, one-container launch, preflight) comes from Tensormesh's
> `netapp-cacheblend`; the CacheBlend plugin is pulled with a Tensormesh API key.

Docs:

- [docs/design.md](docs/design.md): what is measured and how, the design
  decisions, L1 sizing, and why vLLM's GPU prefix cache is off.
- [docs/workload.md](docs/workload.md): the workload, its data, and simulator
  results for the main schedules.
- [docs/troubleshooting.md](docs/troubleshooting.md): error messages and fixes.

**Methodology note:** vLLM's GPU prefix cache is off in all four legs
(`ENABLE_APC=0`), so LMCache's tiers can be measured directly: every reused
token comes from LMCache, by prefix or by CacheBlend, from L1 or Valkey.
LMCache still does prefix caching; each turn reloads its prefix from CPU
memory. Report this with the results ([why](docs/design.md#vllms-gpu-prefix-cache-is-off)).

## Status

What has run on GPUs so far (one H200 NVL host, Ubuntu 25.04, 2026-11-13) and
what has not:

| Path | Status |
|---|---|
| One worker per host, legs A-D, Valkey over TCP (single Valkey 8.1, no auth) | Tested end to end: 4 legs x 72 requests, 0 failures |
| Plugin v0.5.8rc2 on engine v0.5.8rc1, `KV_LOAD_FAILURE_POLICY=recompute`, `L2_PREFETCH_INFLIGHT_WATERMARK=0.2` | Tested (2026-11-16): B and D boot and blend; the runs below had 0 failed KV loads and 0 L1 read-race errors |
| CacheBlend on Qwen3-Coder-30B-A3B, at chunk 256 / 128 / 64, from L1 (B) and with Valkey (D) | Tested (`verify.sh`, `chunk_smoke.sh`: CacheBlend reuses a shifted body) |
| vLLM's GPU prefix cache off; L1 eviction watermark and the free-space check | Tested |
| Simulator against measured legs | Within 3.5 points on the test schedule ([workload.md](docs/workload.md#offline-preview-qwen3-coder-tokenizer-unbounded-valkey-2-workers)) |
| Host Python environment (`scripts/pyenv.sh`) | Tested in Ubuntu 22.04 / 24.04, Amazon Linux 2023 and Rocky 9 containers; not yet on an AWS GPU AMI |
| Valkey over RDMA / EFA (`VALKEY_TRANSPORT=rdma`), `blob-tcp`, the large-object module | **Not tested** (no EFA hardware so far) |
| Two or more workers per host (`WORKERS=2`, the profile's default), worker moves on GPUs | **Not tested** on GPUs (simulator only) |
| Cross-worker CacheBlend (`scripts/coordinator.sh`) | **Not tested** |
| Valkey cluster mode, TLS, password (`VALKEY_CLUSTER`, `VALKEY_TLS`, `VALKEY_PASSWORD`) | **Not tested** |
| 8 x 5 sub-agent schedule (the headline) on one worker: all 40 agents on one GPU, L1 16 GiB at 0.5, warm | Tested: A 63.2 / B 77.2 / C 85.6 / D 92.2% reused, 0 failed requests; D logged 6 L1 allocation failures ([workload.md](docs/workload.md)) |
| The headline on two workers (the profile's setting); 4 x 5 long sessions on GPUs | **Not run yet** (simulated) |
| arm64 hosts | **Not tested** (the engine image is multi-arch) |
| Warm-up cohort before measuring (design decision 8, the default) | Tested: first run and the 8 x 5 sub-agent schedule, warm start |
| Accuracy | Out of scope for this harness: use `lmcache bench quality` |

---

## Layout

```
cacheblend.env        every setting, documented
profiles/momento.env  this engagement's choices (2 workers, Valkey over TCP, L1 16 GiB)
configs/{A,B,C,D}.env the four legs: reuse mode x L2 backend
setup.sh start.sh stop.sh   from netapp-cacheblend; start/stop handle several workers
run_leg.sh            one leg from a clean cache: start workers, replay, collect, stop
scripts/              pyenv.sh (host Python), launch (one worker), entrypoint (in the container), preflight,
                      verify, stage-plugin, smoke_cb.py (does blending fire?), chunk_smoke.sh,
                      sweep.sh (grid of legs),
                      valkey.py (ping/info/count/blob/wipe), coordinator.sh
swarm/                the workload: generate, simulate, replay, summarize
tests/                offline tests and a CPU mock of the serving stack
docs/                 design and methodology, workload, troubleshooting
```

## First run (one GPU, about 45 minutes after setup)

A small schedule through all four legs on one worker, to check a new host end
to end and compare against our reference numbers. You need one GPU with at
least 80 GB (H100 / H200), Docker with the NVIDIA Container Toolkit, driver
>= 580, a Tensormesh API key, about 70 GB of local disk and 20 GB of RAM.

```bash
git clone https://github.com/Tensormesh-Collaboration/momento-cacheblend.git
cd momento-cacheblend
cp cacheblend.env.local.example cacheblend.env.local
cat >> cacheblend.env.local <<'CFG'
DATA_DIR=/data/momento            # local disk: model weights (~61 GB) and caches
WORKERS=1
HOST_GPUS=0
VALKEY_NODES=172.17.0.1:6390      # the throwaway Valkey below; or your own endpoint
CFG

# A throwaway Valkey for C and D, reachable from the worker containers through
# Docker's bridge (skip it if VALKEY_NODES points at a real one)
docker run -d --name momento-valkey -p 172.17.0.1:6390:6379 valkey/valkey:8.1 \
    valkey-server --maxmemory 64gb --maxmemory-policy allkeys-lru --save "" --appendonly no

./setup.sh                 # Python env (.venv), key (asks once), model, preflight, plugin
source .venv/bin/activate

# 2 developers x 3 agents, 12 turns each, 60 s idle gaps: 72 measured requests, prompts up to
# ~25K tokens, plus a warm-up developer (3 agents, 36 requests) replayed first, untimed
python -m swarm.generate --download-shards 1,4 --data data/swesmith --repo getmoto__moto \
    --devs 2 --agents 3 --max-turns 12 --idle-prob 0.3 --idle-s 60 --workers 1 -o schedules/smoke-2x3

# The four legs at 6 GiB of L1 evicting at 50% (about 3 GiB cached: below this
# schedule's 9 GiB working set, so C and D read from Valkey)
EVICTION_WATERMARK=0.5 MAX_REQUEST_TOKENS=26000 \
    scripts/sweep.sh --schedules schedules/smoke-2x3 --legs "A B C D" --l1 6 --name first-run
```

Each leg takes about 8.5 minutes (2 to boot, about 20 s of warm-up, 6 for the
schedule) and starts from empty caches, warmed by the warm-up developer before
measuring. `scripts/sweep.sh` prints one summary table at the end (also in
`runs/<timestamp>-sweep-first-run/summary.md`). Reference, measured on one H200
NVL (dev host, 2026-11-16, plugin v0.5.8rc2):

| Leg | start | reused_pct | non_prefix_pct | l2_share_of_prefix_hits_pct | valkey_evicted_keys | failed_kv_loads |
|---|---|---|---|---|---|---|
| A | warm | 75.0 | 0 | 0 | - | 0 |
| B | warm | 79.0 | 4.6 | - | - | 0 |
| C | warm | 92.7 | 0 | 19.5 | 0 | 0 |
| D | warm | 94.4 | 1.7 | - | 0 | 0 |

From empty caches (`--start cold`, plugin v0.5.8rc1, 2026-11-13 and -15) the
same legs measured 73.8-74.4 / 78.5 / 92.1 / 93.1.

Expect the same within a few points (A to D in that order, C and D well above
A and B), 72 measured requests and 0 failures per leg. Every leg's `verify.sh` must
pass, including "CacheBlend reuses a shifted body" in B and D. If C or D shows
0% reused, check its `w0/lmcache-server.log` for `Failed to batched allocate`
([troubleshooting](docs/troubleshooting.md)).

Clean up: `docker rm -f momento-valkey`.

## 1. The workload (no GPU needed)

```bash
scripts/pyenv.sh            # once: .venv with Python 3.12 (uv; any Linux distribution)
source .venv/bin/activate

# SWE-smith trajectories (MIT), tool split: 8 shards of 80-250 MB; 2 are enough for 4 x 5 on big repos
python -m swarm.generate --download-shards 1,4 --data data/swesmith --repo getmoto__moto \
    --devs 4 --agents 5 --workers 2 -o schedules/moto-4x5

# Sub-agent mode: short-lived agents with a context bundle from their developer
python -m swarm.generate --data data/swesmith --repo getmoto__moto --devs 8 --agents 5 \
    --subagent-turns 6 --bundle-files 4 -o schedules/moto-8x5-sub

# Preview A-D offline: where each input token would come from, per L1 size
python -m swarm.simulate schedules/moto-8x5-sub --l1-gb 16,32 --eviction-watermark 0.5 --fleet-blend
```

Sweeps (design decisions 7, 9, 10):

- **L1 per worker** is the one capacity axis: `--l1-gb 16,24,32`. Keep it
  below the working set so C and D spill to Valkey; A and B evict and
  recompute. The `L2 peak` column is the Valkey capacity the run needed:
  provision more than that, so Valkey never becomes the limit. Pass the same
  `--eviction-watermark` as the legs use (`EVICTION_WATERMARK`, 0.5 in the
  Momento profile): LMCache's L1 caches about watermark x its size, and what
  stays free must hold the largest prompt, or that request stores nothing (the
  simulator's `no-store` column must be 0; on GPUs `validate_config` checks it).
- **Idle-gap eviction pressure**: requests right after an idle gap are grouped
  by how much the agent's old worker stored into L1 while it waited (as a
  fraction of L1), with the share of their tokens still served from cache.
- **Worker-move probability** (optional): `swarm.generate ... --move-prob 0,0.5,1`
  writes `<output>-q0`, `-q0.5`, `-q1`, identical except for which agents move
  after idle gaps (the moves at 0.5 are a subset of those at 1). Pass all three
  to `swarm.simulate`, which accepts several schedules.

Chunk size is a sweep axis too: `--chunk-size 256,128,64` (the default is 256,
CacheBlend's validated size; results and caveats in [docs/workload.md](docs/workload.md)).

Every agent replays one recorded SWE-smith session, teacher-forced; the swarm
around it (developers, harness prompt, preambles, bundles, context clearing,
gaps, idle periods, worker moves) is synthetic and seeded. The schedule
directory is the single input to every leg, to the simulator and to anyone
reproducing the run. Generator knobs: `python -m swarm.generate --help`.

Gaps between turns default to a lognormal (median 4 s, capped at 60 s). To use
measured Claude Code gaps instead:

```bash
python -m swarm.timing semianalysis <cc-traces-weka parquet>... -o data/gaps.json
python -m swarm.generate ... --gap empirical:data/gaps.json
```

## 2. The serving stack

Requirements: Linux (any distribution: `scripts/pyenv.sh` brings its own Python), Docker with the NVIDIA Container Toolkit, driver >= 580,
a Tensormesh API key (`tm_...`), and a Valkey endpoint for C and D. For RDMA:
EFA devices and the EFA software on each GPU host, and the valkey-large-object
module on the Valkey nodes (`scripts/valkey.py blob` checks it). Qwen3-Coder-30B-A3B
is ~61 GB of weights in bf16: one H200 per worker, or `GPUS_PER_WORKER=2` on 80 GB GPUs.

```bash
cp cacheblend.env.local.example cacheblend.env.local
vi cacheblend.env.local     # DATA_DIR, HOST_GPUS, VALKEY_NODES (+ VALKEY_PASSWORD)

./setup.sh                  # once: Python env, key, model download, preflight, plugin staging
./start.sh                  # WORKERS containers; vLLM on :8000+i, LMCache metrics on 127.0.0.1:8080+i
./stop.sh                   # --wipe-l2 also deletes this kit's Valkey keys
```

Each worker is one container running vLLM and the LMCache server (they share
GPU memory over CUDA IPC, so they stay together). Workers on one host share
nothing but the host: an agent that moves between them finds its KV in Valkey
or recomputes it.

Valkey keys live under `VALKEY_KEY_PREFIX@`. `scripts/valkey.py wipe` deletes
only those, never the whole database.

### Valkey over RDMA (EFA)

The profile uses Valkey over TCP because that is the path tested so far
([Status](#status)). Switching to Valkey Large Object over RDMA is a settings
change plus three host prerequisites.

**Prerequisites**

| Where | What | Checked by |
|---|---|---|
| GPU hosts | An EFA-enabled instance type: `/dev/infiniband/uverbs*` present (the kit passes them into each worker) | `preflight.sh` |
| GPU hosts | The AWS EFA software installed at `/opt/amazon/efa` (libfabric >= 2.4). The kit mounts it read-only into each worker (`EFA_HOST_DIR`) | `preflight.sh` (`fi_info -p efa`), the worker's entrypoint (`libfabric.so.1` loads) |
| GPU hosts | Security groups and placement that let EFA traffic reach the Valkey nodes | - |
| Valkey nodes | The valkey-large-object module loaded on every node | `preflight.sh` (`scripts/valkey.py blob`, BLOB.INFO) |
| Engine image | valkey-glide-sync >= 2.6.0: installed into each worker at start (needs network from the container; otherwise bake it into an image) | the worker's entrypoint |

**Settings**, in `cacheblend.env.local`:

```bash
VALKEY_TRANSPORT=rdma
VALKEY_NODES=10.0.1.20:6379,10.0.1.21:6379          # module-enabled nodes
VALKEY_RDMA_INTERFACE=rdmap71s0-rdm,rdmap72s0-rdm   # /opt/amazon/efa/bin/fi_info -p efa | grep domain
EFA_MAX_MR_GB=384                                   # this card's max_mr_size: ibv_devinfo -v | grep max_mr_size
# as the deployment needs:  VALKEY_CLUSTER=1   VALKEY_TLS=1   VALKEY_PASSWORD=...
```

**Check it**, before any measured leg:

```bash
./setup.sh                                  # preflight: EFA devices, fabric domains, registration cap, BLOB.INFO
scripts/chunk_smoke.sh --legs "D" --chunks "256"   # one worker: boots D and checks Valkey receives objects
```

**Limits to know**

- Each RDMA client registers the whole L1 with the card, so `VALKEY_WORKERS x
  L1_SIZE_GB` must fit `EFA_MAX_MR_GB` (`validate_config` refuses otherwise). The
  profile's 4 clients x 16 GiB need 64 GiB.
- With several fabric domains in `VALKEY_RDMA_INTERFACE`, clients are spread over
  them round-robin: keep `VALKEY_WORKERS` a multiple of their number.
- RDMA pins all of L1 when the worker starts (`--no-l1-use-lazy`), so startup takes
  longer with a large L1.
- TCP keys and large-object keys cannot share a key prefix. `run_leg.sh` wipes the
  prefix before every leg; after switching transports by hand, run
  `scripts/valkey.py wipe` first.
- Hosts without EFA can share the same cache with `VALKEY_TRANSPORT=blob-tcp`
  (large-object values over the normal connection; not faster than `tcp`).
- If a worker exits at startup, `docker logs momento-w<i>` names the missing piece
  (libfabric not loadable, no EFA devices in the container, glide too old).

## 3. Run the four legs

The headline run is the sub-agent schedule at the profile's defaults (two
workers, L1 16 GiB at watermark 0.5, Valkey over TCP):

```bash
for leg in A B C D; do ./run_leg.sh $leg --schedule schedules/moto-8x5-sub; done
python -m swarm.summarize runs/*-A runs/*-B runs/*-C runs/*-D --ttft-target 2.0
```

Each leg starts from empty caches and, by default, warms them first: the
schedule's warm-up developer (`swarm.generate --warmup-devs`, 1 by default) is
replayed back to back and untimed, and the leg is measured from when it
finishes, as a fleet that has been running would be (`WARM_START=0`, or
`scripts/sweep.sh --start cold`, measures from empty caches; `--start "warm cold"`
runs both). It then runs the whole schedule and leaves
`runs/<timestamp>-<leg>/` with `requests.jsonl` (per request: TTFT, latency,
tokens, worker, whether it followed an idle gap or a move), `metrics.json`
(LMCache and vLLM counter deltas per worker), `config.env` and each worker's logs.

For grids, `scripts/sweep.sh` runs every cell from empty caches and writes one
summary table, e.g. the optional worker-move sweep:

```bash
scripts/sweep.sh --schedules "schedules/sub-q0 schedules/sub-q0.5 schedules/sub-q1" \
    --legs "A C" --l1 "16 128" --name moves
```

The summary adds, per run: warm or cold start, the sweep coordinates (L1,
chunk, move probability), failed KV loads that were recomputed
(`failed_kv_loads`, from the worker logs, with a warning that also counts
LMCache's L1 read-race and allocation errors), Valkey's own evictions during the leg (`valkey_evicted_keys`,
from `scripts/valkey.py stats` before and after; anything above 0 means Valkey's
capacity limited the run), and an idle-gap table: requests after a gap grouped by
the eviction pressure on the agent's old worker, from the LMCache counters the
replay samples every 5 s (`metrics_timeseries.jsonl`).

For the chunk-size sweep on GPUs, first boot and smoke-test each size
(`scripts/chunk_smoke.sh --legs "B D"`), then run legs with it set, e.g.
`CHUNK_SIZE=64 ./run_leg.sh D --schedule ... --label c64`. Only 256 is validated
for quality; check 128 and 64 with `lmcache bench quality` before reporting them.

`TIME_SCALE` and `IDLE_SCALE` shrink the schedule's gaps for quick runs; the
cache only sees the order and overlap of requests, but eviction during idle
gaps depends on how much other traffic runs in them, so keep `IDLE_SCALE=1`
for the idle-gap sweep.

## Versions

The kit pins the engine image **v0.5.8rc1** (`lmcache/vllm-openai:v0.5.8rc1`,
amd64 + arm64) and the CacheBlend plugin **v0.5.8rc2**. Engine (`ENGINE_TAG`) and
plugin (`TM_VERSION`) must come from the same release line (`validate_config`
checks): the plugin imports LMCache internals that move between releases. Plugin
rc2 is rc1 plus one fix, which only calls v0.5.8rc1 APIs: a CacheBlend request
whose KV load fails is recomputed instead of hanging. There is no v0.5.8rc2
engine image yet.

Known issue in LMCache v0.5.8rc1 (Tensormesh LM-1157): with many requests
prefetching from L2 into a small L1, L1 can fill with data waiting to be used
and stall, and eviction can race with reads and fail KV loads. Seen at 150+
concurrent requests; our test runs (up to 16 per worker) showed neither. The
kit mitigates both: `L2_PREFETCH_INFLIGHT_WATERMARK=0.2` (LMCache's default is
0.5) and `KV_LOAD_FAILURE_POLICY=recompute` in every leg, so a failed load
costs a recompute instead of an HTTP 500. The summary reports how many there
were.

| Needed for | Status in v0.5.8rc1 |
|---|---|
| Valkey over TCP (C, D) | Yes (`VALKEY_TRANSPORT=tcp`) |
| Valkey over RDMA / EFA | Yes (`VALKEY_TRANSPORT=rdma`; untested here, the profile defaults to `tcp`): needs the valkey-large-object module on the servers, valkey-glide-sync >= 2.6.0 in the image (installed at start if missing) and the host's EFA software (libfabric >= 2.4), mounted into each worker |
| Mixed RDMA and non-RDMA hosts on one cache | Yes (`VALKEY_TRANSPORT=blob-tcp` on hosts without EFA) |
| Cross-worker CacheBlend (D with a coordinator) | Yes (`scripts/coordinator.sh`, then `COORDINATOR_URL`) |
| CacheBlend on Qwen3-Coder-30B-A3B (MoE) | The plugin handles routed-expert MoE, but has no card for this model: `verify.sh` runs `scripts/smoke_cb.py`, which fails unless CacheBlend reuses a shifted body |

What changed from v0.5.6rc4 in how the kit launches workers:

- CacheBlend legs use decode-only CUDA graphs (`--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'`)
  instead of `--enforce-eager` (`CB_DECODE_CUDAGRAPH=0` goes back), and the
  prefix legs match them when `FLAG_MATCHED=1`.
- The plugin now coexists with vLLM's GPU prefix cache (byte-identical on dense
  models only). The kit still turns it off in every leg (`ENABLE_APC=0`; see
  the methodology note above).
- The plugin selects the CUSTOM attention backend, block size 64, and turns off
  vLLM's extensible KV cache by itself; the kit passes none of them.
- With RDMA each worker's LMCache server starts with `--no-l1-use-lazy`, and
  `VALKEY_WORKERS x L1_SIZE_GB` must fit the EFA card's registered-memory cap
  (each glide client registers the whole L1). `validate_config` checks it
  against `EFA_MAX_MR_GB`; preflight reads the real cap from `ibv_devinfo`.
  At the Momento profile's 16 GiB, 4 clients register 64 GiB.

## Tests

```bash
.venv/bin/python -m pytest -q tests/
```

Offline: hashing, generator structure and determinism, simulator semantics
(prefix, CacheBlend at another offset, Valkey after a move, LMCache's L1
eviction), metric parsing, and a replay against two mock engines. For what has
run on GPUs, see [Status](#status).
