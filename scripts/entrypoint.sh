#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# scripts/entrypoint.sh — runs INSIDE the container (started by launch.sh).
#
# One container per worker, two processes:
#   1. lmcache server   the KV cache server, L1 (RAM) + optional L2 (Valkey or files)
#   2. vllm serve       the inference engine, connected to it
#
# CACHE_MODE picks what the cache server does:
#   blend    CacheBlend: reuses cached chunks at ANY position in a prompt
#            (--engine-type blend + CBKVConnector)
#   offload  standard LMCache KV offloading: reuses cached PREFIXES only
#            (--engine-type default + LMCacheMPConnector)
# Same container, same L1/L2 tiers either way — only the reuse logic differs.
#
# They talk over 127.0.0.1 and hand KV tensors to each other through CUDA IPC.
# Keeping both in ONE container is deliberate: cross-container CUDA IPC is
# unreliable on some GPUs (notably RTX PRO 6000 Blackwell); inside a single
# container the handoff never crosses a boundary.
#
# All settings arrive as environment variables from cacheblend.env.
# ---------------------------------------------------------------------------
set -Eeuo pipefail

LOG=/var/log/cacheblend
mkdir -p "$LOG"

# --- L2 adapter, built from the L2_* / VALKEY_* knobs ------------------------
L2_ARGS=()
case "${L2_BACKEND:-none}" in
  valkey)
    # The adapter lazy-imports glide_sync: valkey-glide-sync >= 2.3.0 for tcp and
    # blob-tcp, >= 2.6.0 (the first release with the RDMA API) for rdma. Check
    # now, not at the first store.
    GLIDE_MIN=2.3.0; [ "${VALKEY_TRANSPORT:-tcp}" = rdma ] && GLIDE_MIN=2.6.0
    if ! /opt/venv/bin/python - "$GLIDE_MIN" <<'PY' 2>/dev/null
import sys
from importlib.metadata import version
import glide_sync  # noqa: F401
have = tuple(int(x) for x in version("valkey-glide-sync").split(".")[:3] if x.isdigit())
need = tuple(int(x) for x in sys.argv[1].split("."))
sys.exit(0 if have >= need else 1)
PY
    then
      if [ "${VALKEY_PIP_INSTALL:-0}" = 1 ]; then
        # The v0.5.8rc1 image runs as uid 1000 with a root-owned venv and no pip
        # in it, so install into a private directory with uv (shipped in the
        # image) and put it on PYTHONPATH; pip into the venv where that works.
        echo "installing valkey-glide-sync>=${GLIDE_MIN} ..."
        if [ -w /opt/venv/lib ] && /opt/venv/bin/python -m pip --version >/dev/null 2>&1; then
          /opt/venv/bin/python -m pip install -q "valkey-glide-sync>=${GLIDE_MIN}" \
            || { echo "FATAL: pip install valkey-glide-sync failed (no network from the container?)"; exit 1; }
        else
          PYDEPS=/tmp/momento-pydeps
          command -v uv >/dev/null || { echo "FATAL: no writable pip and no uv in the image: bake valkey-glide-sync into it"; exit 1; }
          uv pip install --quiet --python /opt/venv/bin/python --target "${PYDEPS}" "valkey-glide-sync>=${GLIDE_MIN}" \
            || { echo "FATAL: uv pip install valkey-glide-sync failed (no network from the container?)"; exit 1; }
          export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${PYDEPS}"
        fi
        /opt/venv/bin/python -c 'import glide_sync' || { echo "FATAL: valkey-glide-sync installed but not importable"; exit 1; }
      else
        echo "FATAL: the engine image lacks valkey-glide-sync>=${GLIDE_MIN}. Set VALKEY_PIP_INSTALL=1, or bake it into the image."
        exit 1
      fi
    fi
    if [ "${VALKEY_TRANSPORT:-tcp}" = rdma ]; then
      # glide loads libfabric (>= 2.4) at runtime; launch.sh mounts the host's
      # EFA installation at EFA_HOST_DIR and passes the /dev/infiniband devices.
      for d in "${EFA_HOST_DIR:-/opt/amazon/efa}/lib" "${EFA_HOST_DIR:-/opt/amazon/efa}/lib64"; do
        [ -d "$d" ] && export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
      done
      /opt/venv/bin/python -c 'import ctypes; ctypes.CDLL("libfabric.so.1")' 2>/dev/null \
        || { echo "FATAL: libfabric.so.1 not loadable in the container (EFA_HOST_DIR=${EFA_HOST_DIR:-/opt/amazon/efa} mounted? EFA installer on the host?)"; exit 1; }
      ls /dev/infiniband/uverbs* >/dev/null 2>&1 \
        || [ "${VALKEY_RDMA_PROVIDER:-efa-direct}" = tcp ] \
        || { echo "FATAL: no /dev/infiniband/uverbs* in the container: the EFA devices were not passed through"; exit 1; }
    fi
    L2_ADAPTER=$(python3 -c '
import json, os
e = os.environ
cfg = {
    "type": "valkey",
    "startup_nodes": e["VALKEY_NODES"],
    "cluster_mode": e.get("VALKEY_CLUSTER", "0") == "1",
    "tls_enable": e.get("VALKEY_TLS", "0") == "1",
    "num_workers": int(e.get("VALKEY_WORKERS", "8")),
    "request_timeout": float(e.get("VALKEY_REQUEST_TIMEOUT", "5.0")),
}
for env, key in (("VALKEY_USERNAME", "username"), ("VALKEY_PASSWORD", "password"), ("VALKEY_KEY_PREFIX", "key_prefix")):
    if e.get(env):
        cfg[key] = e[env]
if e.get("VALKEY_TTL_S"):
    cfg["ttl_seconds"] = int(e["VALKEY_TTL_S"])
# Only name the transport when it is not the default: releases before
# v0.5.8rc1 reject keys they do not know.
transport = e.get("VALKEY_TRANSPORT", "tcp")
if transport != "tcp":
    cfg["transport"] = transport
if transport == "rdma":
    cfg["rdma_provider"] = e.get("VALKEY_RDMA_PROVIDER", "efa-direct")
    ifs = [x.strip() for x in e.get("VALKEY_RDMA_INTERFACE", "").split(",") if x.strip()]
    if ifs:  # one domain, or a list the workers are spread over round-robin
        cfg["rdma_interface"] = ifs[0] if len(ifs) == 1 else ifs
print(json.dumps(cfg))')
    echo "L2: valkey $(python3 -c 'import json,sys; c=json.loads(sys.argv[1]); c.pop("password",None); print(json.dumps(c))' "${L2_ADAPTER}")"
    ;;
  fs)
    mkdir -p "${L2_CONTAINER_PATH}"
    L2_ADAPTER=$(python3 -c '
import json, os
e = os.environ
cfg = {
    "type": "fs_native",
    "base_path": e["L2_CONTAINER_PATH"],
    "num_workers": int(e.get("L2_WORKERS", "32")),
    "use_odirect": e.get("L2_ODIRECT", "0") == "1",
    "read_ahead_size": int(e.get("L2_READ_AHEAD", "8192")),
}
cap = float(e.get("L2_MAX_GB", "0") or 0)
if cap > 0:  # capped: evict when 80% full, free 20%
    cfg["max_capacity_gb"] = cap
    cfg["eviction"] = {"eviction_policy": e.get("L2_EVICTION", "LRU"),
                       "trigger_watermark": 0.8, "eviction_ratio": 0.2}
print(json.dumps(cfg))')
    echo "L2: files ${L2_ADAPTER}"
    ;;
  none) echo "L2: none (L1 only)" ;;
  *) echo "FATAL: L2_BACKEND='${L2_BACKEND}' must be none, valkey or fs"; exit 1 ;;
esac
if [ -n "${L2_ADAPTER:-}" ]; then
  L2_ARGS=(
    --l2-adapter "${L2_ADAPTER}"
    --l2-store-policy "${L2_STORE_POLICY:-default}"
    --l2-prefetch-policy "${L2_PREFETCH_POLICY:-default}"
    --l2-prefetch-max-in-flight "${L2_PREFETCH_MAX_IN_FLIGHT:-8}"
  )
fi

# --- coordinator (fleet-wide CacheBlend lookup, v0.5.8rc1) ----------------------
COORD_ARGS=()
if [ -n "${COORDINATOR_URL:-}" ]; then
  # Fleet blend lookup is fed by the cache-event stream, so event reporting is on.
  COORD_ARGS=( --coordinator-url "${COORDINATOR_URL}" --coordinator-event-reporting )
  echo "coordinator: ${COORDINATOR_URL} (event reporting on)"
fi

# --- 1. KV cache server ---------------------------------------------------------
# --l1-init-size-gb = --l1-size-gb allocates and pins all of L1 at startup, so
# no pinning happens during serving (a mid-run pin stalls stores).
# L1_INIT_SIZE_GB (when set) pins only that much up front and grows lazily.
# RDMA registers the whole L1 arena with the fabric at start-up, so the arena
# must exist in full: --no-l1-use-lazy (LMCache docs, Valkey RDMA notes).
L1_ARGS=( --l1-init-size-gb "${L1_INIT_SIZE_GB:-${L1_SIZE_GB}}" )
if [ "${L2_BACKEND:-none}" = valkey ] && [ "${VALKEY_TRANSPORT:-tcp}" = rdma ]; then
  L1_ARGS=( --no-l1-use-lazy )
fi
case "${CACHE_MODE:-blend}" in
  blend)   ENGINE_TYPE=blend ;;
  offload) ENGINE_TYPE=default ;;
  *) echo "FATAL: CACHE_MODE='${CACHE_MODE}' must be blend or offload"; exit 1 ;;
esac
echo "=== starting lmcache server: mode=${CACHE_MODE:-blend} (engine ${ENGINE_TYPE}), L1 ${L1_SIZE_GB} GB ==="
/opt/venv/bin/lmcache server \
  --host 127.0.0.1 \
  --port "${LMCACHE_PORT}" \
  --http-host 0.0.0.0 \
  --http-port "${LMCACHE_HTTP_PORT}" \
  --engine-type "${ENGINE_TYPE}" \
  --max-workers "${LMCACHE_MAX_WORKERS}" \
  --worker-reap-timeout-seconds 180 \
  --chunk-size "${CHUNK_SIZE}" \
  --l1-size-gb "${L1_SIZE_GB}" \
  "${L1_ARGS[@]}" \
  --l1-align-bytes "${L1_ALIGN_BYTES}" \
  --eviction-policy LRU \
  --eviction-trigger-watermark "${EVICTION_WATERMARK:-0.8}" \
  --eviction-ratio "${EVICTION_RATIO:-0.2}" \
  ${L2_ARGS[@]+"${L2_ARGS[@]}"} \
  ${COORD_ARGS[@]+"${COORD_ARGS[@]}"} \
  > "$LOG/lmcache-server.log" 2>&1 &
SERVER_PID=$!

# Gate 1 — listening. Up to 300 s: pinning L1 happens before the listen and
# scales with L1_SIZE_GB (~40 s for 200 GB on a typical GPU server).
echo "waiting for the cache server on 127.0.0.1:${LMCACHE_PORT} ..."
for i in $(seq 1 300); do
  if (echo > "/dev/tcp/127.0.0.1/${LMCACHE_PORT}") >/dev/null 2>&1; then
    echo "  listening after ${i}s"; break
  fi
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "FATAL: cache server died on startup"; tail -60 "$LOG/lmcache-server.log"; exit 1; }
  sleep 1
  [ "$i" -eq 300 ] && { echo "FATAL: cache server never opened ${LMCACHE_PORT}"; tail -60 "$LOG/lmcache-server.log"; exit 1; }
done

# Gate 2 — CUDA backend, not the CPU fallback. A server that silently lands on
# the CPU backend cannot open vLLM's CUDA-IPC handles, and the failure only
# appears 300 s later as "did not respond to register_kv_caches". Catch it now.
# (This reads the startup banner: never set LMCACHE_DISABLE_BANNER.)
sleep 5
if grep -qi 'StubCPUDevice\|torch_device_type=cpu' "$LOG/lmcache-server.log"; then
  echo "FATAL: the cache server fell back to the CPU backend."
  echo "       The NVIDIA driver is too old for this image: it needs R580 / CUDA 13."
  grep -i 'cuda\|backend\|device' "$LOG/lmcache-server.log" | tail -20
  exit 1
fi
if grep -qi 'lmcache.c_ops\|CudaPinMemoryBackend\|platform.cuda' "$LOG/lmcache-server.log"; then
  echo "  backend OK (CUDA)"
else
  echo "  WARN: no explicit CUDA backend banner (text varies by version); ./start.sh verifies after startup."
fi

# --- 2. vLLM ------------------------------------------------------------------------
# blend:   what cacheblend-plugin v0.5.8rc1 requires (docs/design.md §7):
#          --no-enable-chunked-prefill (asserted), one whole prompt per step
#          (--max-num-batched-tokens = MAX_MODEL_LEN), and decode-only CUDA
#          graphs (FULL_DECODE_ONLY, mode 0) or --enforce-eager. The plugin
#          picks the CUSTOM attention backend, block size 64 and turns off the
#          extensible KV cache itself. vLLM's GPU prefix cache (APC) coexists
#          with CacheBlend as of v0.5.8rc1; ENABLE_APC decides, for every leg.
# offload: LMCacheMPConnector, prefix reuse only. With FLAG_MATCHED=1 (the
#          default) it runs the same engine flags as blend, so the legs differ
#          only in how LMCache reuses KV; FLAG_MATCHED=0 runs vLLM's production
#          defaults (full CUDA graphs, chunked prefill) for goodput.
KV_TRANSFER_CONFIG=$(python3 -c '
import json, os
e = os.environ
extra = {
    "lmcache.mp.host": "tcp://127.0.0.1",
    "lmcache.mp.port": int(e["LMCACHE_PORT"]),
    "lmcache.mp.mq_timeout": float(e.get("MQ_TIMEOUT", "660")),
}
if e.get("CACHE_MODE", "blend") == "blend":
    connector = "CBKVConnector"
    extra["cb.check_layer"] = int(e.get("CB_CHECK_LAYER", "1"))
    extra["cb.recomp_ratio"] = float(e.get("CB_RECOMP_RATIO", "0.15"))
else:
    connector = "LMCacheMPConnector"
print(json.dumps({"kv_connector": connector, "kv_role": "kv_both",
                  "kv_connector_extra_config": extra}))')
if [ "${CB_DECODE_CUDAGRAPH:-1}" = 1 ]; then
  GRAPH_ARGS=( --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' )
else
  GRAPH_ARGS=( --enforce-eager )
fi
APC_ARGS=( --enable-prefix-caching ); [ "${ENABLE_APC:-1}" = 1 ] || APC_ARGS=( --no-enable-prefix-caching )
if [ "${CACHE_MODE:-blend}" = blend ] || [ "${FLAG_MATCHED:-1}" = 1 ]; then
  MODE_ARGS=( --max-num-seqs "${MAX_NUM_SEQS}"
              --max-num-batched-tokens "${MAX_MODEL_LEN}"
              "${GRAPH_ARGS[@]}"
              --no-enable-chunked-prefill
              "${APC_ARGS[@]}" )
else
  MODE_ARGS=( "${APC_ARGS[@]}" )
fi
# VLLM_EXTRA_ARGS: model-specific flags from cacheblend.env (space-separated).
read -ra EXTRA_ARGS <<< "${VLLM_EXTRA_ARGS:-}"
echo "=== starting vLLM: ${SERVED_NAME}, TP=${TP_SIZE}, ${MODE_ARGS[*]} ${VLLM_EXTRA_ARGS:-} ==="
echo "kv-transfer-config: ${KV_TRANSFER_CONFIG}"

# Log through tee, then exec: vLLM replaces this shell as PID 1, so
# `docker stop` delivers SIGTERM to vLLM and it shuts down cleanly.
exec > >(tee "$LOG/vllm.log") 2>&1
exec /opt/venv/bin/vllm serve "${MODEL}" \
  --served-model-name "${SERVED_NAME}" \
  --host 0.0.0.0 \
  --port "${VLLM_PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  "${MODE_ARGS[@]}" \
  --kv-transfer-config "${KV_TRANSFER_CONFIG}" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
