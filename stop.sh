#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# stop.sh — stop every worker this kit started on this host.
#
#   ./stop.sh               stop the worker containers (L2 is kept)
#   ./stop.sh --wipe-l2     also delete this kit's L2 data: the keys under
#                           VALKEY_KEY_PREFIX in Valkey, or the fs L2 files
# ---------------------------------------------------------------------------
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "${ROOT}/lib/env.sh"; load_env || exit 1

WIPE=0
for a in "$@"; do
  case "$a" in
    --wipe-l2) WIPE=1 ;;
    -h|--help) sed -n '3,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $a (try --help)" >&2; exit 2 ;;
  esac
done

for c in $(docker ps -a --format '{{.Names}}' | grep -E '^momento-w[0-9]+$' || true) cacheblend-download; do
  if [ -n "$(docker ps -aq -f "name=^${c}\$")" ]; then
    remove_ctr "$c" && echo "stopped ${c}"
  fi
done

[ "$WIPE" = 1 ] || { echo "L2 kept (--wipe-l2 to delete it)"; exit 0; }
case "${L2_BACKEND}" in
  valkey) "${PY}" "${KIT_DIR}/scripts/valkey.py" wipe ;;
  fs) wipe_l2 && echo "fs L2 cleared at ${L2_HOST_PATH}" ;;
  none) echo "no L2 configured" ;;
esac
