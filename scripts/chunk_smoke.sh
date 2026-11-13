#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# scripts/chunk_smoke.sh — GPU smoke test of CacheBlend at each chunk size.
#
#   scripts/chunk_smoke.sh                      # legs B, chunks 256 128 64
#   scripts/chunk_smoke.sh --legs "B D" --chunks "256 128 64"
#
# For every (chunk size, leg): one worker on the first GPU slice, empty caches,
# start.sh (which runs verify.sh: backend, KV registration, serving, L2 writes,
# and scripts/smoke_cb.py at that chunk size), then stop. B checks CacheBlend
# from local L1; D adds Valkey, so 12 MiB and 6 MiB objects also go through the
# configured transport (RDMA in the Momento profile).
#
# It answers "does CacheBlend boot and blend at this chunk size on this model",
# not "is it faster or as accurate": throughput comes from run_leg.sh with
# CHUNK_SIZE set, quality from `lmcache bench quality` at each size.
# Output: runs/<timestamp>-chunk-smoke/c<chunk>-<leg>/ (logs, verify output)
# and a summary table at the end.
# ---------------------------------------------------------------------------
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LEGS="B"; CHUNKS="256 128 64"
while [ $# -gt 0 ]; do
  case "$1" in
    --legs) LEGS="${2:?}"; shift 2 ;;
    --chunks) CHUNKS="${2:?}"; shift 2 ;;
    -h|--help) sed -n '3,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

OUT="${ROOT}/runs/$(date +%Y%m%d-%H%M%S)-chunk-smoke"
mkdir -p "$OUT"
declare -a SUMMARY=()
for chunk in $CHUNKS; do
  for leg in $LEGS; do
    dir="${OUT}/c${chunk}-${leg}"; mkdir -p "$dir"
    echo "=== chunk ${chunk}, leg ${leg} ==="
    rc=0
    (
      set -a
      # shellcheck disable=SC1090
      source "${ROOT}/configs/${leg}.env"
      CHUNK_SIZE=${chunk}
      WORKERS=1
      set +a
      # shellcheck source=../lib/env.sh
      source "${ROOT}/lib/env.sh"; load_env || exit 1
      "${KIT_DIR}/stop.sh" >/dev/null
      python3 "${KIT_DIR}/scripts/valkey.py" wipe >/dev/null 2>&1 || true
      "${KIT_DIR}/start.sh"
    ) > "${dir}/start-verify.log" 2>&1 || rc=$?
    cp "${ROOT}"/logs/w0/*.log "$dir/" 2>/dev/null || true
    smoke=$(grep -E '^(PASS|FAIL)  CacheBlend|^FAIL ' "${dir}/start-verify.log" | tail -1 || true)
    reused=$(sed -n 's/^PASS  CacheBlend reused \([0-9]*\) tokens.*/\1/p' "${dir}/start-verify.log" | tail -1)
    status=$([ "$rc" = 0 ] && echo PASS || echo "FAIL (exit ${rc})")
    SUMMARY+=("$(printf '%-6s %-4s %-16s %-10s %s' "$chunk" "$leg" "$status" "${reused:--}" "${smoke:-no smoke result}")")
    tail -5 "${dir}/start-verify.log"
    ( source "${ROOT}/lib/env.sh"; load_env; "${KIT_DIR}/stop.sh" >/dev/null ) || true
  done
done

echo
printf '%-6s %-4s %-16s %-10s %s\n' chunk leg verify reused "smoke line"
printf '%s\n' "${SUMMARY[@]}" | tee "${OUT}/summary.txt"
echo "logs: ${OUT}"
