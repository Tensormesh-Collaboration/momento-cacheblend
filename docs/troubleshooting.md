# Troubleshooting

Every step prints the cause and the fix when it fails. This page collects them, with
background (setup and start rows come from netapp-cacheblend). Logs are in `logs/w<i>/`
per worker: `lmcache-server.log` and `vllm.log`. They are overwritten whenever a worker
starts, so for a benchmark use the copies each leg saves in `runs/<timestamp>-<leg>/w<i>/`.

## Setup

| Message | Cause | Fix |
|---|---|---|
| `artifacts.tensormesh.ai rejected TM_API_KEY ... invalid or expired API key` | Keys expire (30 days by default) or were revoked | Ask the Tensormesh team for a new key, then `sed -i '/^TM_API_KEY=/d' cacheblend.env.local && ./setup.sh` |
| `... is not pullable with this key` | The plugin tag does not exist, or the key is not granted that image | Check `TM_VERSION` (release candidates may not be mirrored for every key); ask the Tensormesh team to grant the image |
| `HF_TOKEN does not start with hf_` | A different token was pasted | `sed -i '/^HF_TOKEN=/d' cacheblend.env.local && ./setup.sh` |
| `the token is valid but this account has NO ACCESS` | The Hugging Face repo is gated | Accept the license on the model page and wait for approval, or set `MODEL_HOST_DIR` to weights you already have |
| `NVIDIA driver ... < 580` | The driver is too old for the CUDA 13 image | Upgrade to R580 or newer |
| `not enough space under DATA_DIR` | The disk cannot hold the model plus L2 | Point `DATA_DIR` at a larger disk, or set `L2_MAX_GB` |
| `cannot create DATA_DIR` | Permissions | `sudo mkdir -p <dir> && sudo chown $USER: <dir>` |
| `another ./setup.sh is already running` | A second copy was started, often in another tmux window | Let the first finish, or stop it |
| Download is very slow | Outbound bandwidth to Hugging Face | Copy the weights over another way and set `MODEL_HOST_DIR` |
| `invalid settings in cacheblend.env` | A setting has the wrong type | Fix the settings listed in the message |

## Start

| Message | Cause | Fix |
|---|---|---|
| `cache server fell back to the CPU backend` | The driver/CUDA combination is below this image's floor | Driver ≥ 580. This check exists to catch the CPU fallback in 5 seconds instead of a 300-second hang. |
| `cache server never opened 555x` | The server could not start, often L1 larger than the RAM it can pin | Lower `L1_SIZE_GB`; the log tail is printed with the error |
| `pin_memory failed ... DMA performance may be degraded` in `lmcache-server.log` (or the benchmark's `L1 pin_memory failed` warning) | One very large startup pin of L1 failed, so KV loads use pageable memory and run several times slower | Set `L1_INIT_SIZE_GB=20` (pin 20 GB up front, grow lazily) or lower `L1_SIZE_GB`, then restart |
| `container momento-w<i> exited during startup` | vLLM failed; the last log lines are printed | Read the printed traceback; most often `MAX_MODEL_LEN` too large for the model, or GPUs already in use |
| Startup stops at `Wrapping ... KV cache tensors` | Waiting on KV registration with the cache server | See below |
| `port is already allocated` | Another service uses 8000 or 8080 | Change `VLLM_PORT` or `LMCACHE_HTTP_PORT` |

### `LMCache server did not respond to register_kv_caches within ...s`

One message with several possible causes. Check in this order:

1. **CPU backend.** `grep -i StubCPUDevice logs/w0/lmcache-server.log`. If it matches, the
   driver is too old. The start script normally catches this.
2. **Version mismatch.** The engine (`ENGINE_TAG`) and plugin (`TM_VERSION`) images must come
   from the same release line (v0.5.8rc1 engine with v0.5.8rc2 plugin is fine). Never use
   `latest` or nightly tags.
3. **Split containers.** `mapping of buffer object failed` means CUDA IPC was refused across
   a container boundary. This repo keeps both processes in one container; make sure nothing
   was changed to split them.
4. **Slow, not stuck.** A large model on many GPUs can register slowly. `MQ_TIMEOUT` is
   already 660 s, more than twice the library default.

## Valkey

| Message | Cause | Fix |
|---|---|---|
| `Valkey at ... does not answer PING` | Wrong `VALKEY_NODES`, a firewall, TLS or auth | `scripts/valkey.py ping` with the same settings; set `VALKEY_TLS=1` or `VALKEY_PASSWORD` as the deployment requires |
| `the engine image has no valkey-glide-sync` | The engine image does not ship the Valkey client | `VALKEY_PIP_INSTALL=1` (the container needs network), or bake `valkey-glide-sync>=2.3.0` into an image |
| L2 objects do not grow in `verify.sh` | Stores fail silently in the server | `grep -i valkey logs/w<i>/lmcache-server.log`; check `maxmemory` and the eviction policy with `scripts/valkey.py info` |
| Adapter refuses `transport` | The engine predates v0.5.8rc1 | `TM_VERSION=v0.5.8rc1 ENGINE_TAG=v0.5.8rc1` (or `VALKEY_TRANSPORT=tcp` on older images) |
| `libfabric.so.1 not loadable in the container` | The host's EFA software is not at `EFA_HOST_DIR` | Install the EFA software on the host, or point `EFA_HOST_DIR` at it |
| `no /dev/infiniband devices` | The EFA driver is not loaded, or the instance has no EFA | Check `fi_info -p efa`; on hosts without EFA use `VALKEY_TRANSPORT=blob-tcp` |
| `... exceeds the EFA registration cap` | Each RDMA client registers the whole L1 | Lower `VALKEY_WORKERS` or `L1_SIZE_GB`; set `EFA_MAX_MR_GB` to this card's `max_mr_size` |
| `needs the valkey-large-object module` | rdma and blob-tcp store module objects | Load `libvalkey_large_object.so` on every node; with ACLs, allow `+@largeobj` |
| Keys stored over tcp do not load over rdma | Plain strings and module objects cannot share a prefix | Wipe the prefix (`scripts/valkey.py wipe`) when switching transports; run_leg.sh does |

## Benchmark

| Message | Cause | Fix |
|---|---|---|
| Replay requests fail with a context-length error | The schedule's longest prompt plus output exceeds `MAX_MODEL_LEN` | `meta.json` lists the prompt-length p90 and max; raise `MAX_MODEL_LEN` or regenerate with `--max-turns` lower |
| A C or D leg starts with L2 hits | Keys from an earlier leg | `run_leg.sh` deletes keys under `VALKEY_KEY_PREFIX` before each leg; by hand: `scripts/valkey.py wipe` |
| `non_prefix_pct` is 0 in a B or D leg | Blending never fired | Run `scripts/smoke_cb.py` against the worker; read the vLLM log's `CB:` lines (backend, block size, degraded scatters) |
| vLLM refuses to boot with CBKVConnector | A flag CacheBlend asserts: chunked prefill on, a compiled mode, `cudagraph_mode=FULL`, PP>1 or speculative decoding | Keep the kit's flags; for an experiment, `CB_DECODE_CUDAGRAPH=0` falls back to `--enforce-eager` |
| D shows no cross-worker blending | No coordinator: each worker matches only its own chunks | `scripts/coordinator.sh start` and set `COORDINATOR_URL` |
| `WARN: Valkey evicted keys during some legs` | Valkey filled up during the leg, so its capacity, not L1, limited the run | Give Valkey more memory than the simulator's `L2 peak` for that schedule, or shrink the swarm; rerun the leg |
| The idle-gap table is empty | No `metrics_timeseries.jsonl` (LMCache metrics unreachable, or `--metrics-interval 0`), no `config.env`, or no idle gaps in the schedule | Run through `run_leg.sh`, on the worker host; check `meta.json`'s `idle_gaps` |
| `LMCache /metrics unreachable` in the summary | The admin port is bound to loopback | Run the replay on the worker host, or tunnel `LMCACHE_HTTP_PORT+i` |
| `entrypoint.sh: ... /var/log/cacheblend/...: Permission denied` | The v0.5.8rc1 engine image runs as uid 1000, not root, and could not write the host's log directory | Fixed in `launch.sh` (it makes the worker log directory and an fs L2 directory writable); for directories you mount yourself, `chmod a+rwx` them |
| `/opt/venv/bin/python: No module named pip` / `pip install valkey-glide-sync failed` | The v0.5.8rc1 image has no pip in its venv, which root owns | Fixed in `entrypoint.sh`: it installs glide with `uv` into a private directory on `PYTHONPATH`. Without network from the container, bake glide into an image |
| A and C show LMCache hits near 0, all reuse as `cached_tokens` | vLLM's GPU prefix cache (`ENABLE_APC=1`) holds more KV than L1 (680K tokens on an H200 with Qwen3-Coder-30B, against 240K in 22 GiB of L1) and serves every repeat itself | `ENABLE_APC=0` for every leg, or cap the GPU cache (`--kv-cache-memory-bytes` in `VLLM_EXTRA_ARGS`) below the swarm's working set |
| A leg on a small L1 stores and loads (almost) nothing; the server log repeats `Failed to batched allocate N memory blocks ... not enough memory` and `L1 memory usage 0.68 below watermark 0.80; skipping eviction` | LMCache evicts only above `EVICTION_WATERMARK`, in a background loop, and allocates a request's chunks all at once: a prompt larger than what is free below the watermark is never stored, and nothing evicts to make room | Keep `(1 - EVICTION_WATERMARK) x L1_SIZE_GB` above the largest prompt (`validate_config` checks it against `MAX_REQUEST_TOKENS`, default `MAX_MODEL_LEN`) |
