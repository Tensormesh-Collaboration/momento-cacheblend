#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# lib/gpu.sh — resolves HOST_GPUS into a docker flag and a GPU count, so the
# rest of the scripts use exactly the GPUs you choose.
#
# Sourced by the scripts that pick GPUs. Not run directly.
#
# HOST_GPUS forms:
#   all        every GPU on the box            -> --gpus all
#   0,1        those specific device ids       -> --gpus '"device=0,1"'
#   3          device id 3 only                -> --gpus '"device=3"'
#   count:2    any 2, driver picks             -> --gpus 2
#
# The count: prefix exists because a bare number is otherwise ambiguous —
# docker reads `--gpus 2` as "two GPUs" but people write HOST_GPUS=2 meaning
# "the third card". Device ids are the default reading; count is opt-in.
# ---------------------------------------------------------------------------

topology_usage() {
  cat >&2 <<'USAGE'
Topology overrides (optional — defaults come from cacheblend.env):
  --gpus <spec>   all | 0,1 | count:N        e.g. --gpus 0,1
  --tp <n>        tensor-parallel size, or   auto
  --model <ref>   container path or HF id
  -h, --help

Default is HOST_GPUS=all (every GPU on the machine).
Use a subset of GPUs:  ./start.sh --gpus 0,1
USAGE
}

# Per-invocation GPU overrides, without editing the env file:
#   ./start.sh --gpus 0,1   ./start.sh --tp 1
parse_topology_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --gpus)    HOST_GPUS="${2:?--gpus needs a value}"; shift 2 ;;
      --gpus=*)  HOST_GPUS="${1#*=}"; shift ;;
      --tp)      TP_SIZE="${2:?--tp needs a value}";    shift 2 ;;
      --tp=*)    TP_SIZE="${1#*=}"; shift ;;
      --model)   MODEL="${2:?--model needs a value}";   shift 2 ;;
      --model=*) MODEL="${1#*=}"; shift ;;
      -h|--help) topology_usage; return 2 ;;
      *) echo "unknown argument: $1" >&2; topology_usage; return 2 ;;
    esac
  done
  return 0
}

resolve_gpu_args() {
  case "${HOST_GPUS:?HOST_GPUS is unset}" in
    all)     GPU_ARGS=( --gpus all ) ;;
    count:*) GPU_ARGS=( --gpus "${HOST_GPUS#count:}" ) ;;
    *[!0-9,]*)
      echo "FATAL: HOST_GPUS='${HOST_GPUS}' is not 'all', 'count:N', or a device list like 0,1" >&2
      return 1 ;;
    *)       GPU_ARGS=( --gpus "\"device=${HOST_GPUS}\"" ) ;;
  esac
}

gpu_count() {
  case "${HOST_GPUS}" in
    all)     nvidia-smi -L 2>/dev/null | grep -c '^GPU ' ;;
    count:*) echo "${HOST_GPUS#count:}" ;;
    *)       awk -F, '{n=0; for(i=1;i<=NF;i++) if($i!="") n++; print n}' <<< "${HOST_GPUS}" ;;
  esac
}

# Which physical device ids will actually be used — for nvidia-smi queries on
# the host. 'all' and 'count:' resolve to every id, since the driver chooses.
gpu_device_list() {
  case "${HOST_GPUS}" in
    all|count:*) nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, - ;;
    *)           echo "${HOST_GPUS}" ;;
  esac
}

resolve_tp() {
  if [ "${TP_SIZE:-auto}" = "auto" ]; then
    TP_RESOLVED=$(gpu_count)
  else
    TP_RESOLVED="${TP_SIZE}"
  fi
  [ "${TP_RESOLVED:-0}" -ge 1 ] 2>/dev/null \
    || { echo "FATAL: could not resolve a tensor-parallel size (got '${TP_RESOLVED}')" >&2; return 1; }
}
