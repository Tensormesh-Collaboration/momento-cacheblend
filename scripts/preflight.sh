#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# scripts/preflight.sh — hard gates, run before a long model load, that catch
# every known setup failure early. Called by ./setup.sh.
# ---------------------------------------------------------------------------
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=../lib/env.sh
source "${ROOT}/lib/env.sh"; load_env || exit 1
# shellcheck source=../lib/gpu.sh
source "${ROOT}/lib/gpu.sh"
parse_topology_args "$@" || exit $?
# Every worker's GPU slice must exist; preflight checks them all together.
SLICES=(); for w in $(worker_ids); do SLICES+=("$(worker_gpus "$w")") || exit 1; done
HOST_GPUS=$(IFS=,; echo "${SLICES[*]}")
resolve_gpu_args || exit 1
TP_SIZE=$([ "${TP_SIZE}" = auto ] && echo "${GPUS_PER_WORKER}" || echo "${TP_SIZE}")
resolve_tp       || exit 1

fail() { echo "FAIL: $*" >&2; exit 1; }
ok()   { echo "  ok  $*"; }

echo "== 1. NVIDIA driver =="
# A node below the cu129 floor makes the cache server silently fall back
# to StubCPUDevice, which then cannot open the engine's CUDA-IPC handles, which
# surfaces 300 s later as "did not respond to register_kv_caches".
# No `| head -1`: with pipefail, closing the pipe early SIGPIPEs nvidia-smi on
# multi-GPU boxes and fails this line intermittently. Take line 1 in bash.
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader); DRV=${DRV%%$'\n'*}; DRV=${DRV%%.*}
[ "${DRV:-0}" -ge 580 ] || fail "driver ${DRV}.x < 580 — need R580 / CUDA 13. Upgrade before continuing."
ok "driver ${DRV}.x"

echo "== 2. Target GPUs present and healthy =="
DEVS=$(gpu_device_list)
echo "     ${WORKERS} worker(s) x ${GPUS_PER_WORKER} GPU(s): devices ${DEVS}, TP=${TP_RESOLVED} per worker"
IFS=',' read -ra G <<< "${DEVS}"
for i in "${G[@]}"; do
  nvidia-smi -i "$i" --query-gpu=name,memory.total,ecc.errors.uncorrected.volatile.total \
    --format=csv,noheader || fail "GPU $i not queryable"
done
ok "${#G[@]} GPU(s) present: ${DEVS}"

echo "== 3. No stale processes holding those GPUs =="
BUSY=$(nvidia-smi -i "${DEVS}" --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$BUSY" -eq 0 ] || fail "$BUSY process(es) already on GPUs ${DEVS}; free them first"
ok "GPUs idle"

echo "== 4. Container GPU view matches HOST_GPUS =="
# --entrypoint: the engine image's default entrypoint is `vllm`, so a bare
# `nvidia-smi` would be passed to vllm as an argument. Capture first, count
# after — a failing `docker run | wc -l` would exit silently under pipefail.
GPU_VIEW=$(docker run --rm "${GPU_ARGS[@]}" --entrypoint nvidia-smi "${ENGINE_IMAGE:?}" \
             --query-gpu=index --format=csv,noheader 2>&1) \
  || fail "could not run nvidia-smi inside ${ENGINE_IMAGE} with ${GPU_ARGS[*]}: ${GPU_VIEW}"
SEEN=$(printf '%s\n' "$GPU_VIEW" | grep -c '^[0-9]' || true)   # grep -c exits 1 on a zero count
EXPECT=$(gpu_count)
[ "$SEEN" -eq "$EXPECT" ] || fail "container saw $SEEN GPUs, expected ${EXPECT} from HOST_GPUS=${HOST_GPUS}"
ok "container sees ${SEEN} GPU(s)"

echo "== 5. Model weights =="
if [ -n "${MODEL_HOST_DIR:-}" ]; then
  [ -d "${MODEL_HOST_DIR}" ] || fail "MODEL_HOST_DIR=${MODEL_HOST_DIR} does not exist. Download once with:
     hf download <repo-id> --local-dir ${MODEL_HOST_DIR}"
  SHARDS=$(find "${MODEL_HOST_DIR}" -maxdepth 2 \( -name '*.safetensors' -o -name '*.bin' \) | wc -l)
  [ "$SHARDS" -gt 0 ] || fail "no weight files under ${MODEL_HOST_DIR} — the download did not complete"
  [ -f "${MODEL_HOST_DIR}/config.json" ] || fail "no config.json under ${MODEL_HOST_DIR}"
  SZ=$(du -sh "${MODEL_HOST_DIR}" | cut -f1)
  ok "${SHARDS} weight file(s), ${SZ}, at ${MODEL_HOST_DIR} (no HF token needed at runtime)"
  # Does it plausibly fit the GPUs we are about to expose?
  WEIGHTS_GB=$(du -sBG "${MODEL_HOST_DIR}" | cut -f1 | tr -dc '0-9')   # cut first: the path has digits too
  # Per worker: one worker's GPUs must hold the whole model.
  TOTAL_VRAM_GB=0
  for i in $(tr ',' ' ' <<< "${SLICES[0]}"); do
    M=$(nvidia-smi -i "$i" --query-gpu=memory.total --format=csv,noheader,nounits)
    TOTAL_VRAM_GB=$(( TOTAL_VRAM_GB + M / 1024 ))
  done
  echo "     weights ~${WEIGHTS_GB}G vs ${TOTAL_VRAM_GB}G VRAM per worker (GPUs ${SLICES[0]})"
  if [ "$WEIGHTS_GB" -ge "$TOTAL_VRAM_GB" ]; then
    fail "weights (~${WEIGHTS_GB}G) do not fit ${TOTAL_VRAM_GB}G of VRAM per worker — raise GPUS_PER_WORKER, use larger GPUs, or the FP8 checkpoint"
  fi
  HEADROOM=$(( TOTAL_VRAM_GB - WEIGHTS_GB ))
  [ "$HEADROOM" -ge 16 ] || echo "  WARN  only ~${HEADROOM}G per worker left for KV and activations; CacheBlend prefills a whole prompt in one step (no chunked prefill), so long prompts may not fit"
  ok "fits with ~${HEADROOM}G headroom for KV"
else
  ok "will pull ${MODEL} from the Hub at startup"
fi

if [ "${L2_BACKEND}" = valkey ]; then
echo "== 6. Valkey L2 =="
python3 "${KIT_DIR}/scripts/valkey.py" ping || fail "Valkey at ${VALKEY_NODES} does not answer (VALKEY_NODES, VALKEY_PASSWORD, VALKEY_TLS)"
python3 "${KIT_DIR}/scripts/valkey.py" info || true
ok "Valkey reachable; this kit's keys live under ${VALKEY_KEY_PREFIX}@"
if [ "${VALKEY_TRANSPORT}" != tcp ]; then
  python3 "${KIT_DIR}/scripts/valkey.py" blob \
    || fail "VALKEY_TRANSPORT=${VALKEY_TRANSPORT} needs the valkey-large-object module on every Valkey node (BLOB.INFO failed)"
  ok "valkey-large-object module answers BLOB.INFO"
fi
if [ "${VALKEY_TRANSPORT}" = rdma ] && [ "${VALKEY_RDMA_PROVIDER}" != tcp ]; then
  ls /dev/infiniband/uverbs* >/dev/null 2>&1 || fail "no /dev/infiniband/uverbs*: the EFA driver is not loaded on this host"
  [ -d "${EFA_HOST_DIR}" ] || fail "EFA_HOST_DIR=${EFA_HOST_DIR} missing: install the EFA software (it provides libfabric >= 2.4)"
  FI_INFO="${EFA_HOST_DIR}/bin/fi_info"; [ -x "$FI_INFO" ] || FI_INFO=$(command -v fi_info || true)
  if [ -n "$FI_INFO" ]; then
    DOMAINS=$("$FI_INFO" -p efa 2>/dev/null | awk '/domain:/{print $2}' | sort -u | paste -sd, -)
    echo "     EFA domains: ${DOMAINS:-none}"
    [ -n "$DOMAINS" ] || fail "fi_info -p efa lists no EFA domains"
    for d in $(tr ',' ' ' <<< "${VALKEY_RDMA_INTERFACE:-}"); do
      [[ ",${DOMAINS}," == *",${d},"* ]] || fail "VALKEY_RDMA_INTERFACE names '${d}', which fi_info -p efa does not list"
    done
  fi
  # Every RDMA client registers the whole L1 arena: VALKEY_WORKERS x L1 must fit
  # the card's max_mr_size. Read the real cap when ibv_devinfo is available.
  if command -v ibv_devinfo >/dev/null 2>&1; then
    MR=$(ibv_devinfo -v 2>/dev/null | awk '/max_mr_size/{print $2; exit}')
    if [ -n "$MR" ]; then
      MR_GB=$(python3 -c 'import sys; print(int(sys.argv[1], 0) // 2**30)' "$MR")   # ibv_devinfo prints hex
      echo "     EFA max_mr_size: ${MR_GB} GiB (EFA_MAX_MR_GB=${EFA_MAX_MR_GB})"
      [ $(( VALKEY_WORKERS * L1_SIZE_GB )) -le "$MR_GB" ] \
        || fail "VALKEY_WORKERS=${VALKEY_WORKERS} x L1_SIZE_GB=${L1_SIZE_GB} GiB exceeds this card's max_mr_size (${MR_GB} GiB)"
    fi
  fi
  ok "EFA ready: ${VALKEY_WORKERS} RDMA clients x ${L1_SIZE_GB} GiB L1 registered per worker"
fi
elif [ "${L2_ENABLED}" = 1 ]; then
echo "== 6. L2 path is real, writable, and on fast storage =="
check_l2_netfs || fail "L2 is not on a network filesystem (L2_REQUIRE_NETFS=1)"
mkdir -p "${L2_HOST_PATH:?}"
touch "${L2_HOST_PATH}/.wtest" && rm -f "${L2_HOST_PATH}/.wtest" || fail "L2 path not writable"
DEV=$(df --output=source "${L2_HOST_PATH}" | tail -1)
FSTYPE=$(findmnt -no FSTYPE -T "${L2_HOST_PATH}" 2>/dev/null || stat -f -c %T "${L2_HOST_PATH}")
echo "     L2 backing: ${DEV} (${FSTYPE})"
case "$FSTYPE" in
  nfs*|lustre|gpfs|cifs|smb*|fuse.*)
    # A shared network filesystem is a supported L2 (one namespace for every
    # node). Print the mount options that decide its throughput.
    OPTS=$(findmnt -no OPTIONS -T "${L2_HOST_PATH}" 2>/dev/null | tr ',' '\n' \
           | grep -E '^(vers|nconnect|rsize|wsize|proto)=' | tr '\n' ' ' || true)
    ok "network filesystem (${FSTYPE}) — shared L2${OPTS:+; ${OPTS}}" ;;
  *)
    case "$DEV" in *nvme*) ok "on NVMe" ;; *) echo "  WARN  ${DEV} does not look like NVMe — L2 will be slow" ;; esac ;;
esac
if [ "${L2_ODIRECT}" = 1 ]; then
  # O_DIRECT can fail on a network filesystem, and the cache server's own
  # O_DIRECT I/O failed on an NFS export until its buffers were aligned to 1 MiB.
  # Check the mount with one write of L1_ALIGN_BYTES, and the alignment.
  if dd if=/dev/zero of="${L2_HOST_PATH}/.odirect-test" bs="${L1_ALIGN_BYTES}" count=2 oflag=direct status=none 2>/dev/null; then
    ok "O_DIRECT writes of ${L1_ALIGN_BYTES} bytes work under ${L2_HOST_PATH}"
  else
    echo "  WARN  an O_DIRECT write failed under ${L2_HOST_PATH}: set L2_ODIRECT=0, or check the mount"
  fi
  rm -f "${L2_HOST_PATH}/.odirect-test"
  FSBS=$(stat -f -c %S "${L2_HOST_PATH}" 2>/dev/null || echo 0)
  if [ "${FSBS:-0}" -le 0 ]; then
    echo "  WARN  could not read the filesystem block size under ${L2_HOST_PATH}"
  elif [ $(( L1_ALIGN_BYTES % FSBS )) -ne 0 ]; then
    echo "  WARN  L1_ALIGN_BYTES=${L1_ALIGN_BYTES} is not a multiple of the filesystem block size ${FSBS}: O_DIRECT I/O may fail"
  else
    ok "L1_ALIGN_BYTES=${L1_ALIGN_BYTES} is a multiple of the filesystem block size (${FSBS})"
  fi
fi
FREE_GB=$(df -BG --output=avail "${L2_HOST_PATH}" | tail -1 | tr -dc '0-9')
[ "$FREE_GB" -ge "${L2_NEED_GB}" ] || fail "only ${FREE_GB}G free under ${L2_HOST_PATH}, need ${L2_NEED_GB}G (L2_MAX_GB, or L2_MIN_FREE_GB when uncapped)"
ok "${FREE_GB}G free"
else
echo "== 6. L2 =="
ok "L2_BACKEND=none — L1 only"
fi

echo "== 7. Host DRAM covers L1 =="
RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
NEED_RAM=$(( WORKERS * L1_SIZE_GB + 64 ))
[ "$RAM_GB" -gt "$NEED_RAM" ] || fail "host has ${RAM_GB}G RAM; ${WORKERS} worker(s) x L1_SIZE_GB=${L1_SIZE_GB} + 64 needs ${NEED_RAM}G"
ok "${RAM_GB}G RAM for ${WORKERS} x ${L1_SIZE_GB}G L1"

echo "== 8. Registry auth =="
registry_login || fail "${REGISTRY} login failed — TM_API_KEY invalid or expired (issue a new one at https://${REGISTRY}/admin/)"
ok "${REGISTRY} authenticated"

echo
echo "PREFLIGHT PASSED"
