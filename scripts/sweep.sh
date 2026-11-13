#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# scripts/sweep.sh — run legs over a grid and summarize them in one table.
#
#   scripts/sweep.sh --schedules "schedules/sub-q0 schedules/sub-q0.5 schedules/sub-q1" \
#                    --legs "C D" --l1 "22" [--chunks "256"] [--name moves]
#
#   --schedules  schedule directories; a worker-move sweep is the -q<p> outputs of
#                `python3 -m swarm.generate ... --move-prob 0,0.5,1` (identical
#                schedules except for which agents move after idle gaps)
#   --legs       any of A B C D (default: all four)
#   --l1         L1 per worker, GiB (design decision 7: the one capacity axis;
#                keep it below the working set so C and D spill to Valkey).
#                With Valkey over RDMA, VALKEY_WORKERS x L1 must still fit the
#                EFA registration cap: validate_config refuses it otherwise.
#   --chunks     chunk sizes (default 256; 128 and 64 are unvalidated for quality)
#
# Each cell is one ./run_leg.sh from empty caches. Results go to
# runs/<timestamp>-sweep-<name>/<schedule>__<leg>__l1<G>__c<C>/ plus summary.md
# and summary.json (python3 -m swarm.summarize over every cell).
# ---------------------------------------------------------------------------
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SCHEDS=""; LEGS="A B C D"; L1S=""; CHUNKS="256"; NAME="grid"; TTFT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --schedules) SCHEDS="${2:?}"; shift 2 ;;
    --legs) LEGS="${2:?}"; shift 2 ;;
    --l1) L1S="${2:?}"; shift 2 ;;
    --chunks) CHUNKS="${2:?}"; shift 2 ;;
    --name) NAME="${2:?}"; shift 2 ;;
    --ttft-target) TTFT="${2:?}"; shift 2 ;;
    -h|--help) sed -n '3,23p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$SCHEDS" ] || { echo "--schedules is required" >&2; exit 2; }
if [ -z "$L1S" ]; then
  L1S=$(bash -c "source '${ROOT}/lib/env.sh'; load_env >/dev/null; echo \${L1_SIZE_GB}")
fi

OUT="${ROOT}/runs/$(date +%Y%m%d-%H%M%S)-sweep-${NAME}"
mkdir -p "$OUT"
cells=(); failed=()
for sched in $SCHEDS; do
  for l1 in $L1S; do
    for chunk in $CHUNKS; do
      for leg in $LEGS; do
        cell="${OUT}/$(basename "$sched")__${leg}__l1${l1}__c${chunk}"
        echo "=== $(basename "$sched")  leg ${leg}  L1 ${l1} GiB  chunk ${chunk}"
        if L1_SIZE_GB="$l1" CHUNK_SIZE="$chunk" "${ROOT}/run_leg.sh" "$leg" --schedule "$sched" --out "$cell"; then
          cells+=("$cell")
        else
          failed+=("$cell"); [ -f "${cell}/requests.jsonl" ] && cells+=("$cell")
        fi
      done
    done
  done
done

if [ ${#cells[@]} -gt 0 ]; then
  ( cd "$ROOT" && python3 -m swarm.summarize "${cells[@]}" ${TTFT:+--ttft-target "$TTFT"} -o "${OUT}/summary.json" ) | tee "${OUT}/summary.md"
fi
[ ${#failed[@]} -eq 0 ] || { printf 'FAILED cells:\n'; printf '  %s\n' "${failed[@]}"; exit 1; }
echo "sweep done: ${OUT}"
