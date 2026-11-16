#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# scripts/launch.sh — start ONE worker's container (called by start.sh).
# Returns as soon as the container is running; the model is still loading.
#
#   scripts/launch.sh --worker <i>
#
# Worker i: container momento-w<i>, GPUs = the i-th slice of HOST_GPUS
# (GPUS_PER_WORKER each), vLLM on VLLM_PORT+i, LMCache admin API and /metrics
# on LMCACHE_HTTP_PORT+i (host loopback), logs in logs/w<i>/.
# Adapted from netapp-cacheblend's launch.sh: L2 can be Valkey, and several
# workers can share a host.
# ---------------------------------------------------------------------------
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=../lib/env.sh
source "${ROOT}/lib/env.sh"; load_env || exit 1

W=""
while [ $# -gt 0 ]; do
  case "$1" in
    --worker) W="${2:?--worker needs an index}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$W" ] || { echo "usage: scripts/launch.sh --worker <i>" >&2; exit 2; }
validate_config || exit 1

NAME=$(worker_name "$W")
PORT=$(worker_port "$W")
HTTP_PORT=$(worker_http_port "$W")
ZMQ_PORT=$(( LMCACHE_PORT + W ))   # distinct, in case DOCKER_EXTRA_ARGS uses --network host
WLOG=$(worker_log_dir "$W")
GPUS=$(worker_gpus "$W") || exit 1
TP=${TP_SIZE}; [ "$TP" = auto ] && TP=${GPUS_PER_WORKER}
MAXW=${LMCACHE_MAX_WORKERS}; [ "$MAXW" = auto ] && MAXW=$TP

echo "worker ${W}: ${NAME}, GPUs ${GPUS} (TP=${TP}), vLLM :${PORT}, LMCache :${HTTP_PORT}"
echo "  mode ${CACHE_MODE}$([ "${CACHE_MODE}" = offload ] && echo " (flag-matched=${FLAG_MATCHED})")   L1 ${L1_SIZE_GB} GB   L2 ${L2_BACKEND}$(
  case "${L2_BACKEND}" in
    valkey) echo " (${VALKEY_NODES}, ${VALKEY_TRANSPORT}$([ "${VALKEY_TRANSPORT}" = rdma ] && echo ", ${VALKEY_WORKERS} clients x ${L1_SIZE_GB} GiB registered"), prefix ${VALKEY_KEY_PREFIX})" ;;
    fs) echo " (${L2_HOST_PATH})" ;;
  esac)$([ -n "${COORDINATOR_URL:-}" ] && echo "   coordinator ${COORDINATOR_URL}")"

# The v0.5.8rc1 engine image runs as uid 1000 (lmcache), not root: directories
# it writes into must be writable by that user, whoever owns them on the host.
mkdir -p "${WLOG}" && chmod a+rwx "${WLOG}"
L2_MOUNT=()
if [ "${L2_BACKEND}" = fs ]; then
  check_l2_netfs || exit 1
  mkdir -p "${L2_HOST_PATH}" && { chmod a+rwx "${L2_HOST_PATH}" 2>/dev/null || true; }
  L2_MOUNT=( -v "${L2_HOST_PATH}:${L2_CONTAINER_PATH}:rw" )
fi

MODEL_ARGS=()
if [ -n "${MODEL_HOST_DIR:-}" ]; then
  [ -d "${MODEL_HOST_DIR}" ] || { echo "FATAL: model directory ${MODEL_HOST_DIR} not found — run ./setup.sh"; exit 1; }
  MODEL_ARGS+=( -v "${MODEL_HOST_DIR}:${MODEL_CONTAINER_PATH}:ro" )
else
  mkdir -p "${HF_CACHE}"
  MODEL_ARGS+=( -v "${HF_CACHE}:/root/.cache/huggingface" ${HF_TOKEN:+-e HF_TOKEN} )
fi

PLUGIN_ENV=()
if [ "${CACHE_MODE}" = blend ]; then
  PLUGIN_ENV=( -v "${PLUGIN_HOST_DIR}:/cbplugin:ro" -e PYTHONPATH=/cbplugin -e VLLM_PLUGINS=lmcache_cacheblend )
fi
read -ra DOCKER_EXTRA <<< "${DOCKER_EXTRA_ARGS:-}"

# Valkey over RDMA: pass the EFA devices through and mount the host's EFA
# installation (libfabric >= 2.4), which glide loads at runtime. The memlock
# limit is already unlimited below, which fabric registration needs.
RDMA_ARGS=()
if [ "${L2_BACKEND}" = valkey ] && [ "${VALKEY_TRANSPORT}" = rdma ]; then
  if [ "${VALKEY_RDMA_PROVIDER}" != tcp ]; then
    for dev in /dev/infiniband/*; do
      [ -e "$dev" ] && RDMA_ARGS+=( --device "$dev" )
    done
    [ ${#RDMA_ARGS[@]} -gt 0 ] || { echo "FATAL: VALKEY_TRANSPORT=rdma but no /dev/infiniband devices on this host (EFA driver loaded?)" >&2; exit 1; }
  fi
  [ -d "${EFA_HOST_DIR}" ] || { echo "FATAL: EFA_HOST_DIR=${EFA_HOST_DIR} not found: install the EFA software (libfabric >= 2.4) or point EFA_HOST_DIR at it" >&2; exit 1; }
  RDMA_ARGS+=( -v "${EFA_HOST_DIR}:${EFA_HOST_DIR}:ro" )
fi
remove_ctr "${NAME}"

# Secrets reach the container by NAME (-e VAR), never as values on the command
# line, which every user can read with ps.
export VALKEY_PASSWORD="${VALKEY_PASSWORD:-}"
docker run -d --name "${NAME}" \
  --gpus "\"device=${GPUS}\"" \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --shm-size=32g --ipc=private \
  -p "${PORT}:${PORT}" \
  -p "127.0.0.1:${HTTP_PORT}:${HTTP_PORT}" \
  ${L2_MOUNT[@]+"${L2_MOUNT[@]}"} \
  "${MODEL_ARGS[@]}" \
  -v "${CFG_DIR}:/cfg:ro" \
  -v "${WLOG}:/var/log/cacheblend" \
  ${PLUGIN_ENV[@]+"${PLUGIN_ENV[@]}"} \
  ${RDMA_ARGS[@]+"${RDMA_ARGS[@]}"} \
  ${DOCKER_EXTRA[@]+"${DOCKER_EXTRA[@]}"} \
  -e PYTHONHASHSEED=0 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e LMCACHE_LOG_LEVEL=DEBUG \
  -e MODEL="${MODEL}" \
  -e SERVED_NAME="${SERVED_NAME}" \
  -e TP_SIZE="${TP}" \
  -e MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  -e MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
  -e GPU_MEM_UTIL="${GPU_MEM_UTIL}" \
  -e VLLM_PORT="${PORT}" \
  -e LMCACHE_PORT="${ZMQ_PORT}" \
  -e LMCACHE_HTTP_PORT="${HTTP_PORT}" \
  -e LMCACHE_MAX_WORKERS="${MAXW}" \
  -e MQ_TIMEOUT="${MQ_TIMEOUT}" \
  -e CHUNK_SIZE="${CHUNK_SIZE}" \
  -e CB_CHECK_LAYER="${CB_CHECK_LAYER}" \
  -e CB_RECOMP_RATIO="${CB_RECOMP_RATIO}" \
  -e L1_SIZE_GB="${L1_SIZE_GB}" \
  -e L1_INIT_SIZE_GB="${L1_INIT_SIZE_GB:-}" \
  -e L1_ALIGN_BYTES="${L1_ALIGN_BYTES}" \
  -e EVICTION_WATERMARK="${EVICTION_WATERMARK:-0.8}" \
  -e EVICTION_RATIO="${EVICTION_RATIO:-0.2}" \
  -e CACHE_MODE="${CACHE_MODE}" \
  -e FLAG_MATCHED="${FLAG_MATCHED}" \
  -e CB_DECODE_CUDAGRAPH="${CB_DECODE_CUDAGRAPH}" \
  -e ENABLE_APC="${ENABLE_APC}" \
  -e VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}" \
  -e L2_BACKEND="${L2_BACKEND}" \
  -e L2_CONTAINER_PATH="${L2_CONTAINER_PATH}" \
  -e L2_MAX_GB="${L2_MAX_GB}" \
  -e L2_EVICTION="${L2_EVICTION}" \
  -e L2_WORKERS="${L2_WORKERS}" \
  -e L2_ODIRECT="${L2_ODIRECT}" \
  -e L2_READ_AHEAD="${L2_READ_AHEAD}" \
  -e L2_STORE_POLICY="${L2_STORE_POLICY}" \
  -e L2_PREFETCH_POLICY="${L2_PREFETCH_POLICY}" \
  -e L2_PREFETCH_INFLIGHT_WATERMARK="${L2_PREFETCH_INFLIGHT_WATERMARK}" \
  -e KV_LOAD_FAILURE_POLICY="${KV_LOAD_FAILURE_POLICY}" \
  -e VALKEY_NODES="${VALKEY_NODES}" \
  -e VALKEY_CLUSTER="${VALKEY_CLUSTER}" \
  -e VALKEY_TLS="${VALKEY_TLS}" \
  -e VALKEY_USERNAME="${VALKEY_USERNAME:-}" \
  -e VALKEY_PASSWORD \
  -e VALKEY_KEY_PREFIX="${VALKEY_KEY_PREFIX}" \
  -e VALKEY_WORKERS="${VALKEY_WORKERS}" \
  -e VALKEY_REQUEST_TIMEOUT="${VALKEY_REQUEST_TIMEOUT}" \
  -e VALKEY_TTL_S="${VALKEY_TTL_S:-}" \
  -e VALKEY_TRANSPORT="${VALKEY_TRANSPORT}" \
  -e VALKEY_RDMA_PROVIDER="${VALKEY_RDMA_PROVIDER}" \
  -e VALKEY_RDMA_INTERFACE="${VALKEY_RDMA_INTERFACE:-}" \
  -e VALKEY_PIP_INSTALL="${VALKEY_PIP_INSTALL}" \
  -e EFA_HOST_DIR="${EFA_HOST_DIR}" \
  -e COORDINATOR_URL="${COORDINATOR_URL:-}" \
  --entrypoint /bin/bash \
  "${ENGINE_IMAGE}" /cfg/scripts/entrypoint.sh >/dev/null

echo "  container ${NAME} started (follow: docker logs -f ${NAME})"
