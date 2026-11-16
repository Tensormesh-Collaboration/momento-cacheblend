#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_leg.sh — run one configuration (leg) of the benchmark on this host's
# workers, from a clean cache, and keep everything needed to compare legs.
#
#   ./run_leg.sh <A|B|C|D> --schedule schedules/moto-4x5 [--label L] [--out DIR] [--keep-running]
#
#   A  prefix reuse, local L1          C  prefix reuse, L1 + Valkey
#   B  CacheBlend, local L1            D  CacheBlend, L1 + Valkey
#
# Steps: stop the workers, delete this kit's Valkey keys (and fs L2), start the
# workers in the leg's mode (configs/<leg>.env), replay the schedule against
# them, copy each worker's logs into the run folder, stop the workers.
# Results: runs/<timestamp>-<leg>[-label]/  (requests.jsonl, metrics.json,
# run.json, w<i>/ logs, config.env). Compare legs with:
#   python3 -m swarm.summarize runs/*-A runs/*-B runs/*-C runs/*-D
#
# All workers must be on this host. For workers on several hosts, start them
# there with the same configs/<leg>.env and run `python3 -m swarm.replay`
# from one place with every engine URL.
# ---------------------------------------------------------------------------
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEG=${1:-}; shift || true
case "$LEG" in A|B|C|D) ;; *) sed -n '3,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2 ;; esac
LABEL=""; KEEP=0; SCHED=""; OUT_ARG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --schedule) SCHED="${2:?}"; shift 2 ;;
    --label) LABEL="-${2:?}"; shift 2 ;;
    --keep-running) KEEP=1; shift ;;
    --out) OUT_ARG="${2:?}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# The leg's overlay wins over every settings file: export it before load_env,
# which never overrides variables already set in the environment.
set -a
# shellcheck disable=SC1090
source "${ROOT}/configs/${LEG}.env"
set +a
# shellcheck source=lib/env.sh
source "${ROOT}/lib/env.sh"; load_env || exit 1
need_pyenv "run_leg.sh (the replay client)" || exit 1
SCHED=${SCHED:-$SCHEDULE}
[ -f "${SCHED}/sessions.jsonl" ] || { echo "FATAL: no schedule at '${SCHED}' (python -m swarm.generate ... -o <dir>)" >&2; exit 1; }
if [ "$LEG" = D ] && [ -z "${COORDINATOR_URL:-}" ] && [ "${WORKERS}" -gt 1 ]; then
  echo "NOTE: D without COORDINATOR_URL: each worker blends only chunks it stored itself (no cross-worker blending)."
fi

OUT=${OUT_ARG:-"${KIT_DIR}/runs/$(date +%Y%m%d-%H%M%S)-${LEG}${LABEL}"}
mkdir -p "$OUT"
echo "=== leg ${LEG}: ${CACHE_MODE} + L2 ${L2_BACKEND}, ${WORKERS} worker(s), schedule ${SCHED} -> ${OUT}"

"${KIT_DIR}/stop.sh" >/dev/null
# Clean caches: Valkey keys under our prefix go whatever this leg's L2 is, so
# a C or D leg never starts warm from an earlier leg.
"${PY}" "${KIT_DIR}/scripts/valkey.py" wipe 2>/dev/null || true
[ "${L2_BACKEND}" = fs ] && wipe_l2
"${KIT_DIR}/start.sh"

ENGINES=""; METRICS=""
for w in $(worker_ids); do
  ENGINES+="${ENGINES:+,}http://127.0.0.1:$(worker_port "$w")"
  METRICS+="${METRICS:+,}http://127.0.0.1:$(worker_http_port "$w")/metrics"
done
# Every setting, exported or not (load_env sets most without exporting them).
for v in $(compgen -v | grep -E '^(CACHE_MODE|L2_|VALKEY_|L1_|CHUNK_SIZE|WORKERS|GPUS_PER_WORKER|HOST_GPUS|GPU_MEM_UTIL|TM_VERSION|ENGINE_TAG|HF_REPO|MAX_MODEL_LEN|MAX_NUM_SEQS|FLAG_MATCHED|ENABLE_APC|CB_|COORDINATOR_URL|TIME_SCALE|IDLE_SCALE|EVICTION_|MAX_REQUEST_TOKENS|WARM_START|KV_LOAD_FAILURE_POLICY)' \
         | grep -v PASSWORD | sort); do
  printf '%s=%s\n' "$v" "${!v}"
done > "${OUT}/config.env"

# Valkey's own counters around the replay: its evicted_keys must not grow, or
# Valkey's capacity, not L1, limited the leg (design decision 7).
if [ "${L2_BACKEND}" = valkey ]; then
  "${PY}" "${KIT_DIR}/scripts/valkey.py" stats > "${OUT}/valkey_before.json" || rm -f "${OUT}/valkey_before.json"
fi
rc=0
( cd "${KIT_DIR}" && "${PY}" -m swarm.replay "${SCHED}" --engines "${ENGINES}" --lmcache-metrics "${METRICS}" \
    --time-scale "${TIME_SCALE}" --idle-scale "${IDLE_SCALE}" $([ "${WARM_START:-1}" = 1 ] || echo --cold) \
    -o "${OUT}" ) || rc=$?

if [ "${L2_BACKEND}" = valkey ]; then
  "${PY}" "${KIT_DIR}/scripts/valkey.py" stats > "${OUT}/valkey_after.json" || rm -f "${OUT}/valkey_after.json"
fi
for w in $(worker_ids); do
  mkdir -p "${OUT}/w${w}"
  cp "$(worker_log_dir "$w")"/*.log "${OUT}/w${w}/" 2>/dev/null || true
done
[ "$KEEP" = 1 ] || "${KIT_DIR}/stop.sh" >/dev/null
( cd "${KIT_DIR}" && "${PY}" -m swarm.summarize "${OUT}" --ttft-target "${TTFT_TARGET}" ) || true
echo "=== leg ${LEG} done (replay exit ${rc}): ${OUT}"
exit "$rc"
