# CacheBlend + Valkey: agent-swarm benchmark (Momento × Tensormesh)

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

Design notes, data choices and open decisions:
[Momento × Tensormesh agent-swarm benchmark (doc)](https://claude.ai/artifact/7pfdfPvNXfesxdZduGu3rR).
Short version in [docs/workload.md](docs/workload.md).

---

## Layout

```
cacheblend.env        every setting, documented
profiles/momento.env  this engagement's choices (2 workers, Valkey over TCP)
configs/{A,B,C,D}.env the four legs: reuse mode x L2 backend
setup.sh start.sh stop.sh   from netapp-cacheblend; start/stop handle several workers
run_leg.sh            one leg from a clean cache: start workers, replay, collect, stop
scripts/              launch (one worker), entrypoint (in the container), preflight,
                      verify, stage-plugin, smoke_cb.py (does blending fire?), chunk_smoke.sh,
                      valkey.py (ping/info/count/blob/wipe), coordinator.sh
swarm/                the workload: generate, simulate, replay, summarize
tests/                offline tests and a CPU mock of the serving stack
docs/                 workload design, troubleshooting
```

## 1. The workload (no GPU needed)

```bash
pip install -r requirements.txt

# SWE-smith trajectories (MIT), tool split: 8 shards of 80-250 MB; 2 are enough for 4 x 5 on big repos
python3 -m swarm.generate --download-shards 1,4 --data data/swesmith --repo getmoto__moto \
    --devs 4 --agents 5 --workers 2 -o schedules/moto-4x5

# Sub-agent mode: short-lived agents with a context bundle from their developer
python3 -m swarm.generate --data data/swesmith --repo getmoto__moto --devs 8 --agents 5 \
    --subagent-turns 6 --bundle-files 4 -o schedules/moto-8x5-sub

# Preview A-D offline: where each input token would come from, per L1 size
python3 -m swarm.simulate schedules/moto-8x5-sub --l1-gb 8,32 --fleet-blend
```

Chunk size is a sweep axis too: `--chunk-size 256,128,64` (the default is 256,
CacheBlend's validated size; results and caveats in [docs/workload.md](docs/workload.md)).

Every agent replays one recorded SWE-smith session, teacher-forced; the swarm
around it (developers, harness prompt, preambles, bundles, context clearing,
gaps, idle periods, worker moves) is synthetic and seeded. The schedule
directory is the single input to every leg, to the simulator and to anyone
reproducing the run. Generator knobs: `python3 -m swarm.generate --help`.

Gaps between turns default to a lognormal (median 4 s, capped at 60 s). To use
measured Claude Code gaps instead:

```bash
python3 -m swarm.timing semianalysis <cc-traces-weka parquet>... -o data/gaps.json
python3 -m swarm.generate ... --gap empirical:data/gaps.json
```

## 2. The serving stack

Requirements: Linux, Docker with the NVIDIA Container Toolkit, driver ≥ 580,
a Tensormesh API key (`tm_…`), and a Valkey endpoint for C and D. For RDMA:
EFA devices and the EFA software on each GPU host, and the valkey-large-object
module on the Valkey nodes (`scripts/valkey.py blob` checks it). Qwen3-Coder-30B-A3B
is ~61 GB of weights in bf16: one H200 per worker, or `GPUS_PER_WORKER=2` on 80 GB GPUs.

```bash
cp cacheblend.env.local.example cacheblend.env.local
vi cacheblend.env.local     # DATA_DIR, HOST_GPUS, VALKEY_NODES (+ VALKEY_PASSWORD)

./setup.sh                  # once: key, model download, preflight, plugin staging
./start.sh                  # WORKERS containers; vLLM on :8000+i, LMCache metrics on 127.0.0.1:8080+i
./stop.sh                   # --wipe-l2 also deletes this kit's Valkey keys
```

Each worker is one container running vLLM and the LMCache server (they share
GPU memory over CUDA IPC, so they stay together). Workers on one host share
nothing but the host: an agent that moves between them finds its KV in Valkey
or recomputes it.

Valkey keys live under `VALKEY_KEY_PREFIX@`. `scripts/valkey.py wipe` deletes
only those, never the whole database.

## 3. Run the four legs

```bash
for leg in A B C D; do ./run_leg.sh $leg --schedule schedules/moto-4x5; done
python3 -m swarm.summarize runs/*-A runs/*-B runs/*-C runs/*-D --ttft-target 2.0
```

Each leg starts from empty caches, runs the whole schedule, and leaves
`runs/<timestamp>-<leg>/` with `requests.jsonl` (per request: TTFT, latency,
tokens, worker, whether it followed an idle gap or a move), `metrics.json`
(LMCache and vLLM counter deltas per worker), `config.env` and each worker's logs.

For the chunk-size sweep on GPUs, first boot and smoke-test each size
(`scripts/chunk_smoke.sh --legs "B D"`), then run legs with it set, e.g.
`CHUNK_SIZE=64 ./run_leg.sh D --schedule ... --label c64`. Only 256 is validated
for quality; check 128 and 64 with `lmcache bench quality` before reporting them.

`TIME_SCALE` and `IDLE_SCALE` shrink the schedule's gaps for quick runs; the
cache only sees the order and overlap of requests, but eviction during idle
gaps depends on how much other traffic runs in them, so keep `IDLE_SCALE=1`
for the idle-gap sweep.

## Versions

The kit pins **v0.5.8rc1** for both the engine image (`lmcache/vllm-openai:v0.5.8rc1`,
amd64 + arm64) and the CacheBlend plugin. Engine (`ENGINE_TAG`) and plugin
(`TM_VERSION`) must always be the same release.

| Needed for | Status in v0.5.8rc1 |
|---|---|
| Valkey over TCP (C, D) | Yes (`VALKEY_TRANSPORT=tcp`) |
| Valkey over RDMA / EFA | Yes (`VALKEY_TRANSPORT=rdma`, the Momento profile's default): needs the valkey-large-object module on the servers, valkey-glide-sync ≥ 2.6.0 in the image (installed at start if missing) and the host's EFA software (libfabric ≥ 2.4), mounted into each worker |
| Mixed RDMA and non-RDMA hosts on one cache | Yes (`VALKEY_TRANSPORT=blob-tcp` on hosts without EFA) |
| Cross-worker CacheBlend (D with a coordinator) | Yes (`scripts/coordinator.sh`, then `COORDINATOR_URL`) |
| CacheBlend on Qwen3-Coder-30B-A3B (MoE) | The plugin handles routed-expert MoE, but has no card for this model: `verify.sh` runs `scripts/smoke_cb.py`, which fails unless CacheBlend reuses a shifted body |

What changed from v0.5.6rc4 in how the kit launches workers:

- CacheBlend legs use decode-only CUDA graphs (`--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'`)
  instead of `--enforce-eager` (`CB_DECODE_CUDAGRAPH=0` goes back), and the
  prefix legs match them when `FLAG_MATCHED=1`.
- vLLM's GPU prefix cache is on in every leg (`ENABLE_APC=1`): the plugin now
  coexists with it. Validated byte-identical on dense models only.
- The plugin selects the CUSTOM attention backend, block size 64, and turns off
  vLLM's extensible KV cache by itself; the kit passes none of them.
- With RDMA each worker's LMCache server starts with `--no-l1-use-lazy`, and
  `VALKEY_WORKERS × L1_SIZE_GB` must fit the EFA card's registered-memory cap
  (each glide client registers the whole L1). `validate_config` checks it
  against `EFA_MAX_MR_GB`; preflight reads the real cap from `ibv_devinfo`.
  The Momento profile uses 4 clients × 22 GiB.

## Tests

```bash
python3 -m pytest -q tests/
```

Offline: hashing, generator structure and determinism, simulator semantics
(prefix, CacheBlend at another offset, Valkey after a move, LRU eviction), and a
replay against two mock engines. The shell scripts have not been run on GPUs yet.
