#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# start.sh — bring up every worker on this host and prove each is healthy.
#
#   ./start.sh                 # WORKERS containers, GPUs split from HOST_GPUS
#   ./start.sh --restart       # replace running workers
#
# Each worker is one container (vLLM + LMCache server). All are launched first,
# so their model loads overlap, then each is health-checked (CUDA backend, KV
# registration, serving, and L2 writes when L2 is on).
# ---------------------------------------------------------------------------
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "${ROOT}/lib/env.sh"; load_env || exit 1

RESTART=0
for a in "$@"; do
  case "$a" in
    --restart) RESTART=1 ;;
    -h|--help) sed -n '3,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $a (try --help)" >&2; exit 2 ;;
  esac
done

if [ "${CACHE_MODE}" = blend ] && [ ! -d "${PLUGIN_HOST_DIR}/lmcache_cacheblend" ]; then
  echo "FATAL: plugin not staged. Run ./setup.sh first." >&2; exit 1
fi
if [ "${L2_BACKEND}" = valkey ]; then
  "${PY}" "${KIT_DIR}/scripts/valkey.py" ping >/dev/null \
    || { echo "FATAL: Valkey at ${VALKEY_NODES} does not answer PING (scripts/valkey.py ping)" >&2; exit 1; }
fi

launched=()
for w in $(worker_ids); do
  name=$(worker_name "$w"); port=$(worker_port "$w")
  if [ "$RESTART" -eq 0 ] && ctr_running "$name" && curl -sf "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
    echo "worker ${w} already serving on :${port} (./start.sh --restart to replace)"
    continue
  fi
  "${KIT_DIR}/scripts/launch.sh" --worker "$w"
  launched+=("$w")
done

for w in ${launched[@]+"${launched[@]}"}; do
  wait_for_api "$(worker_name "$w")" "$(worker_port "$w")"
  "${KIT_DIR}/scripts/verify.sh" --worker "$w"
done

echo
echo "Workers UP (model: ${SERVED_NAME}):"
for w in $(worker_ids); do
  echo "  worker ${w}: http://$(hostname -f 2>/dev/null || hostname):$(worker_port "$w")/v1   metrics http://127.0.0.1:$(worker_http_port "$w")/metrics"
done
echo "  run a leg: ./run_leg.sh <A|B|C|D> --schedule <dir>      stop: ./stop.sh"
