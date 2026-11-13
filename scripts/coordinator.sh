#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# scripts/coordinator.sh — run the LMCache MP coordinator with fleet-wide
# CacheBlend lookup, for config D across workers.
#
#   scripts/coordinator.sh start     # container momento-coordinator on :COORDINATOR_PORT
#   scripts/coordinator.sh stop
#
# Then set COORDINATOR_URL=http://<this host>:<port> in cacheblend.env.local on
# every worker host and restart the workers. Requires an engine image whose
# LMCache has fleet blend lookup (v0.5.8rc1, the default here); on v0.5.6rc4
# the flag is unknown and the coordinator refuses to start.
#
# With a coordinator, each worker's CacheBlend lookup asks the coordinator
# instead of its local matcher, so a worker can blend a chunk another worker
# stored, fetched from the shared Valkey L2.
# ---------------------------------------------------------------------------
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=../lib/env.sh
source "${ROOT}/lib/env.sh"; load_env || exit 1
NAME=momento-coordinator

case "${1:-}" in
  start)
    remove_ctr "$NAME"
    docker run -d --name "$NAME" -p "${COORDINATOR_PORT}:${COORDINATOR_PORT}" \
      --entrypoint /opt/venv/bin/lmcache "${ENGINE_IMAGE}" \
      coordinator --host 0.0.0.0 --port "${COORDINATOR_PORT}" \
      --enable-blend-lookup --chunk-size "${CHUNK_SIZE}" >/dev/null
    sleep 5
    if ! ctr_running "$NAME"; then
      echo "FATAL: coordinator exited. Last lines:" >&2
      docker logs --tail 40 "$NAME" >&2 || true
      echo "(--enable-blend-lookup needs LMCache v0.5.8rc1 or later in ${ENGINE_IMAGE})" >&2
      exit 1
    fi
    echo "coordinator up: set COORDINATOR_URL=http://$(hostname -I 2>/dev/null | awk '{print $1}'):${COORDINATOR_PORT} on every worker host"
    ;;
  stop) remove_ctr "$NAME" && echo "stopped ${NAME}" ;;
  *) sed -n '3,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
