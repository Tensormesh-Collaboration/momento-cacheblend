# ---------------------------------------------------------------------------
# lib/env.sh — loads the config so every script can be run bare, from any
# directory. Sourced, never executed.
#
#   cacheblend.env         every setting, documented
#   profiles/<name>.env    optional, selected by CB_PROFILE: a named set of
#                          overrides for one deployment (profiles/momento.env)
#   cacheblend.env.local   optional, gitignored: your secrets and overrides
#                          (setup.sh writes TM_API_KEY / HF_TOKEN here)
#
# Precedence: a variable already set in the caller's environment wins, then
# .local, then the profile, then the template. So `HOST_GPUS=0,1 ./start.sh`
# works as expected.
#
# Paths the operator never needs to think about are derived here, after both
# files are read, so overriding DATA_DIR in .local moves everything with it.
# ---------------------------------------------------------------------------

# Repository root: this file lives in <root>/lib/.
KIT_DIR=${KIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

load_env() {
  [ -n "${CACHEBLEND_ENV_LOADED:-}" ] && return 0
  local base="${KIT_DIR}/cacheblend.env" local_f="${KIT_DIR}/cacheblend.env.local"
  [ -f "$base" ] || { echo "FATAL: ${base} missing" >&2; return 1; }

  # Optional profile (profiles/<name>.env), layered between the template and
  # .local: CB_PROFILE from the caller's environment, else the last
  # CB_PROFILE= line in .local or the template. Empty = no profile.
  local profile="${CB_PROFILE-}"
  if [ -z "${CB_PROFILE+x}" ]; then
    local src=("$base"); [ -f "$local_f" ] && src+=("$local_f")
    profile=$(grep -hoE "^CB_PROFILE=[\"']?[A-Za-z0-9_.-]*" "${src[@]}" 2>/dev/null | tail -1 | cut -d= -f2 | tr -d "\"'" || true)
  fi
  local profile_f=""
  if [ -n "$profile" ]; then
    profile_f="${KIT_DIR}/profiles/${profile}.env"
    [ -f "$profile_f" ] || { echo "FATAL: CB_PROFILE=${profile} but ${profile_f} is missing" >&2; return 1; }
  fi

  # Remember anything the caller set explicitly, so the files cannot clobber it.
  local files=("$base") n kept=""
  [ -n "$profile_f" ] && files+=("$profile_f")
  [ -f "$local_f" ] && files+=("$local_f")
  for n in $(grep -hoE '^[A-Za-z_][A-Za-z0-9_]*=' "${files[@]}" | tr -d = | sort -u); do
    if [ -n "${!n+x}" ]; then
      printf -v "__keep_${n}" '%s' "${!n}"; kept="${kept} ${n}"
    fi
  done

  set -a
  # shellcheck disable=SC1090
  for f in "${files[@]}"; do source "$f"; done
  for n in $kept; do
    local k="__keep_${n}"; printf -v "$n" '%s' "${!k}"; unset "$k"
  done

  # --- derived paths (override any of them in cacheblend.env.local) --------
  CFG_DIR=${CFG_DIR:-$KIT_DIR}
  PLUGIN_HOST_DIR=${PLUGIN_HOST_DIR:-$CFG_DIR/cbplugin}
  LOG_DIR=${LOG_DIR:-$CFG_DIR/logs}
  MODEL_HOST_DIR=${MODEL_HOST_DIR-$DATA_DIR/models/$SERVED_NAME}
  L2_HOST_PATH=${L2_HOST_PATH:-$DATA_DIR/lmcache-l2}
  HF_CACHE=${HF_CACHE:-$DATA_DIR/hf-cache}
  L2_CONTAINER_PATH=${L2_CONTAINER_PATH:-/data/lmcache/l2}
  # L2_ENABLED keeps its NetApp-kit meaning, "a filesystem L2 is in use", so the
  # file-L2 helpers below still apply; Valkey has its own checks.
  L2_ENABLED=$([ "${L2_BACKEND:-none}" = fs ] && echo 1 || echo 0)
  # Disk L2 needs: its cap when capped, else the configured free-space floor.
  if [ "${L2_ENABLED:-1}" = 1 ]; then
    L2_NEED_GB=$([ "${L2_MAX_GB:-0}" != 0 ] && echo "${L2_MAX_GB}" || echo "${L2_MIN_FREE_GB:-450}")
  else
    L2_NEED_GB=0
  fi
  CACHEBLEND_ENV_LOADED=1
  set +a
}

# ctr_running <name> — true if that container is running. Tests docker's output
# instead of piping into `grep -q`, which can SIGPIPE docker and, under
# pipefail, report a running container as stopped.
ctr_running() { [ -n "$(docker ps -q -f "name=^$1\$" 2>/dev/null)" ]; }

# remove_ctr <name> [timeout_s] — remove a container and wait until it is gone.
# A single `docker rm -f` is not enough: while the cache server unpins a large
# L1 the container's PID is a zombie for a while, docker refuses to kill it,
# and the next launch finds the name, port and GPUs still taken.
remove_ctr() {
  local name=$1 timeout=${2:-300} t=0
  while [ -n "$(docker ps -aq -f "name=^${name}\$" 2>/dev/null)" ]; do
    if [ "$t" -ge "$timeout" ]; then
      echo "FATAL: container ${name} still present after ${timeout}s (docker rm -f ${name})" >&2
      return 1
    fi
    docker rm -f "${name}" >/dev/null 2>&1 || true
    sleep 5; t=$((t + 5))
  done
}

# --- workers ---------------------------------------------------------------
# Worker i (WORKER_BASE <= i < WORKER_BASE + WORKERS) is container momento-w<i>,
# serving vLLM on VLLM_PORT+i and LMCache's admin API on LMCACHE_HTTP_PORT+i,
# with logs under logs/w<i>/.
worker_ids()       { seq "${WORKER_BASE:-0}" $(( ${WORKER_BASE:-0} + ${WORKERS:-1} - 1 )); }
worker_name()      { echo "momento-w$1"; }
worker_port()      { echo $(( VLLM_PORT + $1 )); }
worker_http_port() { echo $(( LMCACHE_HTTP_PORT + $1 )); }
worker_log_dir()   { echo "${LOG_DIR}/w$1"; }

# worker_gpus <i> — the HOST_GPUS slice for worker i: GPUS_PER_WORKER devices,
# taken in order from HOST_GPUS (a device list, or every GPU for 'all').
worker_gpus() {
  local i=$(( $1 - ${WORKER_BASE:-0} )) n=${GPUS_PER_WORKER:-1} list
  case "${HOST_GPUS}" in
    all) list=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, -) ;;
    count:*) echo "FATAL: HOST_GPUS=count:N cannot be split across workers; list device ids" >&2; return 1 ;;
    *) list=${HOST_GPUS} ;;
  esac
  local -a g; IFS=',' read -ra g <<< "$list"
  if [ $(( (i + 1) * n )) -gt ${#g[@]} ]; then
    echo "FATAL: worker $1 needs GPUs $(( i * n ))..$(( (i + 1) * n - 1 )) of HOST_GPUS=${list} (WORKERS=${WORKERS} x GPUS_PER_WORKER=${n})" >&2
    return 1
  fi
  local IFS=,; echo "${g[*]:$(( i * n )):$n}"
}

# wait_for_api <container> [port] [timeout_s] — block until vLLM answers, or
# explain why it never will. Dies loudly if the container exits during startup.
wait_for_api() {
  local ctr=$1 port=${2:-$VLLM_PORT} timeout=${3:-3600} t=0
  echo "waiting for vLLM in ${ctr} on :${port} (model load takes several minutes) ..."
  until curl -sf "http://localhost:${port}/v1/models" >/dev/null 2>&1; do
    if ! ctr_running "$ctr"; then
      echo "FATAL: container ${ctr} exited during startup. Last lines:" >&2
      docker logs --tail 80 "$ctr" 2>&1 | sed 's/^/  | /' >&2
      echo "Diagnosis: docs/troubleshooting.md (for a register_kv_caches timeout, check the backend banner first)." >&2
      return 1
    fi
    [ "$t" -ge "$timeout" ] && { echo "FATAL: not serving after ${timeout}s" >&2; return 1; }
    [ $((t % 60)) -eq 0 ] && [ "$t" -gt 0 ] && echo "  still loading (${t}s) — last: $(docker logs --tail 1 "$ctr" 2>&1 | cut -c1-120)"
    sleep 10; t=$((t + 10))
  done
  echo "  serving after ~${t}s"
}

# L2 directory ownership. The kit deletes everything under L2_HOST_PATH between
# benchmark legs, which on a shared export is only safe in a directory the kit
# owns. It claims a directory by writing this marker, and only into an empty
# one: launch.sh refuses a non-empty directory without it, wipe_l2 never
# deletes anything without it.
L2_MARKER=.cacheblend-l2

# l2_fstype — filesystem type at L2_HOST_PATH, or at its nearest existing parent.
l2_fstype() {
  local p=${L2_HOST_PATH%/}
  while [ -n "$p" ] && [ ! -e "$p" ]; do p=$(dirname "$p"); done
  findmnt -no FSTYPE -T "${p:-/}" 2>/dev/null || stat -f -c %T "${p:-/}" 2>/dev/null || true
}

# check_l2_netfs — with L2_REQUIRE_NETFS=1, refuse an L2 that is not on a
# network filesystem. Catches an export that is not mounted (after a reboot or
# an automount timeout), where the L2 directory would otherwise be created on
# the local disk and the run would quietly measure that instead.
check_l2_netfs() {
  [ "${L2_ENABLED:-1}" = 1 ] && [ "${L2_REQUIRE_NETFS:-0}" = 1 ] || return 0
  local fs; fs=$(l2_fstype)
  case "$fs" in nfs*|lustre|gpfs|cifs|smb*|fuse.*) return 0 ;; esac
  echo "FATAL: L2_REQUIRE_NETFS=1 but ${L2_HOST_PATH} is on '${fs:-unknown}', not a network filesystem." >&2
  echo "       Is the export mounted?  findmnt -T ${L2_HOST_PATH}" >&2
  echo "       (For a local-disk L2, set L2_REQUIRE_NETFS=0 in cacheblend.env.local.)" >&2
  return 1
}

# wipe_l2 — empty the L2 directory. The container writes it as root, so do the
# delete inside a throwaway container rather than asking the operator for sudo.
# Deletes only in a directory carrying the kit's marker, never crosses into
# another filesystem mounted below it, and keeps the marker.
wipe_l2() {
  local p=${L2_HOST_PATH%/} rc=0
  [ -d "${p:?L2_HOST_PATH is empty}" ] || return 0
  case "$p" in /|/home|/root|/mnt|/data|"${DATA_DIR%/}") echo "FATAL: refusing to wipe L2_HOST_PATH=${L2_HOST_PATH}" >&2; return 1 ;; esac
  docker run --rm -v "${p}:/l2" --entrypoint /bin/sh "${ENGINE_IMAGE}" -c '
    m=/l2/'"${L2_MARKER}"'
    if [ ! -f "$m" ]; then [ -z "$(ls -A /l2)" ] && exit 0; exit 3; fi
    find /l2 -xdev -mindepth 1 ! -path "$m" -delete' || rc=$?
  if [ "$rc" = 3 ]; then
    echo "FATAL: refusing to wipe ${p}: it is not empty and has no ${L2_MARKER} marker, so this kit did not create it." >&2
    echo "       Point L2_HOST_PATH at a dedicated (empty) subdirectory. If this directory really is" >&2
    echo "       only this kit's L2 cache:  touch ${p}/${L2_MARKER}" >&2
    return 1
  fi
  return "$rc"
}

# registry_login — docker login to the Tensormesh artifacts service with the
# tm_ key. Key goes on stdin (never argv). On failure prints the server's reason.
registry_login() {
  local out
  out=$(printf '%s' "${TM_API_KEY:?TM_API_KEY is empty}" \
        | docker login "${REGISTRY:?}" -u x-access-token --password-stdin 2>&1) && return 0
  printf '%s\n' "$out" | grep -viE 'warning|credential|credentials-store|trying next endpoint' | tail -1 >&2 || true
  return 1
}

# model_complete <dir> — true if <dir> holds a usable checkpoint: config.json
# plus every shard its safetensors index names (or at least one shard when
# there is no index). Lets setup accept weights the operator already has.
model_complete() {
  local d=$1
  [ -f "$d/config.json" ] || return 1
  if [ -f "$d/model.safetensors.index.json" ]; then
    python3 -c 'import json,os,sys; d=sys.argv[1]; s=set(json.load(open(os.path.join(d,"model.safetensors.index.json")))["weight_map"].values()); sys.exit(0 if all(os.path.isfile(os.path.join(d,f)) for f in s) else 1)' "$d" 2>/dev/null
  else
    compgen -G "$d/*.safetensors" >/dev/null
  fi
}

# validate_config — catch a mistyped knob before it becomes a confusing
# failure minutes into startup. Prints every problem, returns 1 if any.
validate_config() {
  local errs=() v
  for v in L1_SIZE_GB L1_ALIGN_BYTES L2_MAX_GB L2_MIN_FREE_GB L2_WORKERS L2_READ_AHEAD L2_PREFETCH_MAX_IN_FLIGHT \
           MAX_MODEL_LEN MAX_NUM_SEQS VLLM_PORT LMCACHE_PORT LMCACHE_HTTP_PORT \
           CHUNK_SIZE CB_CHECK_LAYER MODEL_DISK_GB WORKERS GPUS_PER_WORKER WORKER_BASE VALKEY_WORKERS; do
    [[ "${!v:-}" =~ ^[0-9]+$ ]] || errs+=("$v='${!v:-}' must be a whole number")
  done
  for v in GPU_MEM_UTIL CB_RECOMP_RATIO MQ_TIMEOUT; do
    [[ "${!v:-}" =~ ^[0-9]+(\.[0-9]+)?$ ]] || errs+=("$v='${!v:-}' must be a number")
  done
  for v in L2_ODIRECT L2_REQUIRE_NETFS FLAG_MATCHED VALKEY_CLUSTER VALKEY_TLS VALKEY_PIP_INSTALL CB_DECODE_CUDAGRAPH ENABLE_APC; do
    [[ "${!v:-}" =~ ^[01]$ ]] || errs+=("$v='${!v:-}' must be 0 or 1")
  done
  [[ "${L1_INIT_SIZE_GB:-}" =~ ^[0-9]*$ ]] || errs+=("L1_INIT_SIZE_GB='${L1_INIT_SIZE_GB}' must be empty or a whole number")
  if [[ "${L1_INIT_SIZE_GB:-}" =~ ^[0-9]+$ ]] && [[ "${L1_SIZE_GB:-}" =~ ^[0-9]+$ ]] && [ "${L1_INIT_SIZE_GB}" -gt "${L1_SIZE_GB}" ]; then
    errs+=("L1_INIT_SIZE_GB=${L1_INIT_SIZE_GB} is larger than L1_SIZE_GB=${L1_SIZE_GB}")
  fi
  [[ "${LMCACHE_MAX_WORKERS:-}" =~ ^(auto|[0-9]+)$ ]] || errs+=("LMCACHE_MAX_WORKERS='${LMCACHE_MAX_WORKERS:-}' must be auto or a whole number")
  case "${CACHE_MODE:-}" in blend|offload) ;; *) errs+=("CACHE_MODE='${CACHE_MODE:-}' must be blend or offload") ;; esac
  case "${L2_STORE_POLICY:-}" in default|skip_l1) ;; *) errs+=("L2_STORE_POLICY='${L2_STORE_POLICY:-}' must be default or skip_l1") ;; esac
  case "${L2_PREFETCH_POLICY:-}" in default|retain) ;; *) errs+=("L2_PREFETCH_POLICY='${L2_PREFETCH_POLICY:-}' must be default or retain") ;; esac
  case "${L2_EVICTION:-}" in LRU) ;; *) errs+=("L2_EVICTION='${L2_EVICTION:-}' must be LRU (the only policy in this LMCache version)") ;; esac
  case "${L2_BACKEND:-}" in none|valkey|fs) ;; *) errs+=("L2_BACKEND='${L2_BACKEND:-}' must be none, valkey or fs") ;; esac
  case "${VALKEY_TRANSPORT:-}" in tcp|rdma|blob-tcp) ;; *) errs+=("VALKEY_TRANSPORT='${VALKEY_TRANSPORT:-}' must be tcp, rdma or blob-tcp") ;; esac
  if [ "${L2_BACKEND:-}" = valkey ] && [ "${VALKEY_TRANSPORT:-}" = rdma ] \
     && [[ "${VALKEY_WORKERS:-}" =~ ^[0-9]+$ ]] && [[ "${EFA_MAX_MR_GB:-}" =~ ^[0-9]+$ ]] && [[ "${L1_SIZE_GB:-}" =~ ^[0-9]+$ ]]; then
    # Every RDMA client registers the whole L1 arena with the fabric
    # (LMCache docs: "num_workers times the L1 size must fit the device's cap").
    if [ $(( VALKEY_WORKERS * L1_SIZE_GB )) -gt "${EFA_MAX_MR_GB}" ]; then
      errs+=("VALKEY_WORKERS=${VALKEY_WORKERS} x L1_SIZE_GB=${L1_SIZE_GB} = $(( VALKEY_WORKERS * L1_SIZE_GB )) GiB exceeds the EFA registration cap EFA_MAX_MR_GB=${EFA_MAX_MR_GB}: lower either (or set EFA_MAX_MR_GB from ibv_devinfo -v)")
    fi
    if [ -n "${VALKEY_RDMA_INTERFACE:-}" ]; then
      n_if=$(awk -F, '{print NF}' <<< "${VALKEY_RDMA_INTERFACE}")
      [ $(( VALKEY_WORKERS % n_if )) -eq 0 ] || errs+=("VALKEY_WORKERS=${VALKEY_WORKERS} is not a multiple of the ${n_if} RDMA interfaces in VALKEY_RDMA_INTERFACE")
    fi
  fi
  case "${VALKEY_KEY_PREFIX:-}" in *@*) errs+=("VALKEY_KEY_PREFIX must not contain '@'") ;; esac
  [[ "${VALKEY_TTL_S:-}" =~ ^[0-9]*$ ]] || errs+=("VALKEY_TTL_S='${VALKEY_TTL_S}' must be empty or a whole number")
  if [ "${L2_BACKEND:-}" = valkey ] && [ -z "${VALKEY_NODES:-}" ]; then errs+=("L2_BACKEND=valkey needs VALKEY_NODES=host:port"); fi
  if [[ "${L1_ALIGN_BYTES:-}" =~ ^[0-9]+$ ]] && [ $(( 25165824 % L1_ALIGN_BYTES )) -ne 0 ] && [[ "${HF_REPO:-}" == *Qwen3-Coder-30B* ]]; then
    errs+=("L1_ALIGN_BYTES=${L1_ALIGN_BYTES} does not divide Qwen3-Coder-30B's 24 MiB chunk: every chunk would be padded (use 8388608)")
  fi
  if [ "${L2_ODIRECT:-0}" = 1 ] && [[ "${L1_ALIGN_BYTES:-0}" =~ ^[0-9]+$ ]] && [ "${L1_ALIGN_BYTES}" -lt 4096 ]; then
    errs+=("L2_ODIRECT=1 needs L1_ALIGN_BYTES >= 4096")
  fi
  [ "${CHUNK_SIZE:-256}" = 256 ] || errs+=("CHUNK_SIZE must stay 256 (CacheBlend's validated reuse granularity)")
  if [ ${#errs[@]} -gt 0 ]; then
    echo "FATAL: invalid settings in cacheblend.env / profiles / cacheblend.env.local:" >&2
    printf '  - %s\n' "${errs[@]}" >&2
    return 1
  fi
}
