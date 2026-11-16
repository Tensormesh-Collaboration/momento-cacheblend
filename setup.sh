#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup.sh — one-time setup. Run once per machine; safe to re-run, since every
# step skips work that is already done.
#
#   ./setup.sh                 # all GPUs (default)
#   ./setup.sh --gpus 0,1      # only GPUs 0 and 1
#
# Asks for the Tensormesh API key (and a Hugging Face token only if the model
# must be downloaded from a gated repo), downloads the model, runs the 8 preflight gates, and
# stages the CacheBlend plugin. Next step: ./start.sh
# (From netapp-cacheblend, unchanged apart from the worker containers' names.)
# ---------------------------------------------------------------------------
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "${ROOT}/lib/env.sh"; load_env || exit 1
# shellcheck source=lib/gpu.sh
source "${ROOT}/lib/gpu.sh"
parse_topology_args "$@" || exit $?
validate_config || exit 1

LOCAL_ENV="${KIT_DIR}/cacheblend.env.local"

# One setup at a time: two concurrent runs start two downloaders into the same
# directory, which fight over Hugging Face's file locks and crawl.
exec 9>"${KIT_DIR}/.setup.lock"
if command -v flock >/dev/null 2>&1; then   # util-linux; absent on macOS
  flock -n 9 || { echo "FATAL: another ./setup.sh is already running (check your other tmux windows)." >&2; exit 1; }
fi
step() { echo; echo "=== [$1/6] $2 ==="; }
die()  { echo "FATAL: $*" >&2; exit 1; }

# Ask once, remember in the gitignored .local file, never echo the value.
remember_secret() {  # remember_secret <VAR> <prompt>
  local var=$1 val
  [ -t 0 ] || die "$var is empty and there is no terminal to ask on. Add $var=... to ${LOCAL_ENV}"
  read -rsp "$2: " val; echo
  [ -n "$val" ] || die "$var is required"
  touch "$LOCAL_ENV"; chmod 600 "$LOCAL_ENV"
  grep -v "^${var}=" "$LOCAL_ENV" > "${LOCAL_ENV}.tmp" || true
  echo "${var}=${val}" >> "${LOCAL_ENV}.tmp"; mv "${LOCAL_ENV}.tmp" "$LOCAL_ENV"
  printf -v "$var" '%s' "$val"; export "${var?}"
  echo "  saved to cacheblend.env.local"
}

# ---------------------------------------------------------------------------
step 1 "Host Python environment (.venv, for the replay client and swarm tools)"
if "${KIT_DIR}/.venv/bin/python" -c 'import numpy, pyarrow, openai, transformers' >/dev/null 2>&1; then
  echo "  ${KIT_DIR}/.venv ok ($("${KIT_DIR}/.venv/bin/python" --version))"
else
  "${KIT_DIR}/scripts/pyenv.sh"
fi
PY="${KIT_DIR}/.venv/bin/python"   # lib/env.sh picked python3 before .venv existed

# ---------------------------------------------------------------------------
step 2 "Registry access (${REGISTRY})"
[ -n "${TM_API_KEY:-}" ] || remember_secret TM_API_KEY "Tensormesh API key, starts with tm_ (TM_API_KEY)"
case "${TM_API_KEY}" in tm_*) ;; *) die "TM_API_KEY does not start with tm_. Replace it:  sed -i '/^TM_API_KEY=/d' ${LOCAL_ENV} && ./setup.sh" ;; esac
registry_login || die "${REGISTRY} rejected TM_API_KEY (reason above). Keys expire and can be revoked:
       issue a new one at https://${REGISTRY}/admin/ (Customers -> Issue Key), then
       sed -i '/^TM_API_KEY=/d' ${LOCAL_ENV} && ./setup.sh     (it will ask again)"
echo "  ${REGISTRY} ok"
# Checking the exact tag now, not after a 20-minute setup, is the whole point:
# engine and plugin must come from the same release line (a documented cause of
# the 300 s hang; validate_config checks it).
docker manifest inspect "${PLUGIN_IMAGE}" >/dev/null 2>&1 \
  || die "${PLUGIN_IMAGE} is not pullable with this key — the tag may not exist, or the key's image grants exclude it."
echo "  ${PLUGIN_IMAGE} available"

# ---------------------------------------------------------------------------
step 3 "Driver, storage, engine image"
# Checked here as well as in preflight so a too-old driver fails BEFORE a
# 110 GB download, not after it.
echo "  querying driver (nvidia-smi) ..."
# Capture ALL of nvidia-smi's output, then take line 1 in bash. Piping into
# `head -1` closes the pipe early; nvidia-smi (one line per GPU) then dies of
# SIGPIPE and pipefail turns a good answer into a failure — intermittently.
DRV_RAW=$(timeout 60 nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1) \
  || die "nvidia-smi failed or hung for 60s: '${DRV_RAW:-}'. Run nvidia-smi by hand; the GPUs or driver may be in a bad state."
DRV=${DRV_RAW%%$'\n'*}; DRV=${DRV%%.*}
case "$DRV" in ''|*[!0-9]*) die "could not read the driver version from nvidia-smi (got '${DRV}')" ;; esac
[ "${DRV:-0}" -ge 580 ] || die "NVIDIA driver ${DRV}.x < 580. CacheBlend needs R580 / CUDA 13 — upgrade the driver first."
echo "  driver ${DRV}.x ok"
mkdir -p "${DATA_DIR}" "${HF_CACHE}" 2>/dev/null \
  || die "cannot create ${DATA_DIR}. Once, as an admin: sudo mkdir -p ${DATA_DIR} && sudo chown $(id -un): ${DATA_DIR}"
FREE_GB=$(df -BG --output=avail "${DATA_DIR}" | tail -1 | tr -dc '0-9')
HAVE_MODEL=0; { [ -z "${MODEL_HOST_DIR:-}" ] || model_complete "${MODEL_HOST_DIR}"; } && HAVE_MODEL=1
# L2 counts against DATA_DIR only when it sits on the same filesystem (the
# default, $DATA_DIR/lmcache-l2). A shared L2 elsewhere, such as an NFS export,
# is sized by preflight gate 6 against its own free space instead.
L2_ON_DATA=0
if [ "${L2_ENABLED}" = 1 ]; then
  l2_probe=${L2_HOST_PATH}
  while [ ! -e "$l2_probe" ] && [ "$l2_probe" != / ]; do l2_probe=$(dirname "$l2_probe"); done
  [ "$(df --output=target "$l2_probe" | tail -1)" = "$(df --output=target "${DATA_DIR}" | tail -1)" ] && L2_ON_DATA=1
fi
NEED_GB=$(( (L2_ON_DATA ? L2_NEED_GB : 0) + (HAVE_MODEL ? 0 : MODEL_DISK_GB) ))
L2_NOTE=$([ "$L2_ON_DATA" = 1 ] && echo "L2 ${L2_NEED_GB}G" || echo "L2 not on this disk")
echo "  data dir ${DATA_DIR}: ${FREE_GB}G free, need ~${NEED_GB}G (${L2_NOTE}$([ $HAVE_MODEL = 1 ] || echo " + model ${MODEL_DISK_GB}G"))"
[ "$FREE_GB" -ge "$NEED_GB" ] || die "not enough space under ${DATA_DIR}. Pick a bigger disk:
       echo 'DATA_DIR=/mnt/<disk>/cacheblend' >> ${LOCAL_ENV}    (see: df -h)"
docker pull -q "${ENGINE_IMAGE}" >/dev/null && echo "  ${ENGINE_IMAGE} ok"

# ---------------------------------------------------------------------------
step 4 "Model weights"
if [ -z "${MODEL_HOST_DIR:-}" ]; then
  echo "  MODEL_HOST_DIR is empty: vLLM pulls ${MODEL} from the Hub at start. Skipping."
elif [ -f "${MODEL_HOST_DIR}/.download-complete" ] || model_complete "${MODEL_HOST_DIR}"; then
  echo "  model already at ${MODEL_HOST_DIR} — no download needed"
else
  mkdir -p "${MODEL_HOST_DIR}" 2>/dev/null || die "cannot create ${MODEL_HOST_DIR}"
  DL=cacheblend-download
  download() {
    # The engine image already ships huggingface_hub, so nothing is installed
    # on the host. snapshot_download resumes a partial download.
    # Named, so a leftover from an interrupted run is replaced rather than left
    # racing this one; removed on Ctrl-C, since killing `docker run` alone
    # leaves the container running. HF_TOKEN is passed by NAME only — a value
    # on the command line is readable by every user via `ps`.
    docker rm -f "$DL" >/dev/null 2>&1 || true
    trap 'docker rm -f "$DL" >/dev/null 2>&1; exit 130' INT TERM
    docker run --rm --name "$DL" \
      -v "${MODEL_HOST_DIR}:/out" -v "${HF_CACHE}:/root/.cache/huggingface" \
      ${HF_TOKEN:+-e HF_TOKEN} -e HF_HUB_ENABLE_HF_TRANSFER=0 \
      --entrypoint /opt/venv/bin/python "${ENGINE_IMAGE}" -c \
      "from huggingface_hub import snapshot_download as s; s('${HF_REPO}', local_dir='/out')"
  }
  # Settle access BEFORE a 110 GB download. huggingface_hub does not fail
  # cleanly on a bad token: it can fall back to "returning existing local_dir"
  # and exit 0 with half a model, so ask the Hub directly and name the problem.
  hf_code() {  # hf_code <url> [token] -> HTTP status. Token goes via stdin, not argv (ps).
    # `if`, not `&&`: a false `&&` would fail the pipeline under pipefail.
    { if [ -n "${2:-}" ]; then printf 'header = "Authorization: Bearer %s"\n' "$2"; fi; } \
      | curl -sL -K - -o /dev/null -w '%{http_code}' --max-time 30 "$1" || true   # -L: some repos answer via a redirect (307); curl -w prints 000 on failure
  }
  probe="https://huggingface.co/${HF_REPO}/resolve/main/config.json"
  if [ -z "${HF_TOKEN:-}" ]; then
    case "$(hf_code "$probe")" in
      200) ;;                                   # public repo: no token needed
      401|403) echo "  ${HF_REPO} is gated on Hugging Face — a token is needed."
               remember_secret HF_TOKEN "Hugging Face read token, starts with hf_ (HF_TOKEN)" ;;
      000) die "cannot reach huggingface.co from this machine (proxy/firewall?)" ;;
      *) die "unexpected answer from huggingface.co checking ${HF_REPO}: HTTP $(hf_code "$probe")" ;;
    esac
  fi
  if [ -n "${HF_TOKEN:-}" ]; then
    replace="Replace it:  sed -i '/^HF_TOKEN=/d' ${LOCAL_ENV} && ./setup.sh"
    case "${HF_TOKEN}" in hf_*) ;; *) die "HF_TOKEN does not start with hf_ — that is not a Hugging Face token. ${replace}" ;; esac
    [ "$(hf_code https://huggingface.co/api/whoami-v2 "$HF_TOKEN")" = 200 ] \
      || die "Hugging Face rejected HF_TOKEN (invalid or revoked). Create a Read token at https://huggingface.co/settings/tokens. ${replace}"
    case "$(hf_code "$probe" "$HF_TOKEN")" in
      200) echo "  Hugging Face token ok, access to ${HF_REPO} confirmed" ;;
      *) die "the token is valid but this account has NO ACCESS to ${HF_REPO}.
       Open https://huggingface.co/${HF_REPO} while logged in, accept the license, and wait
       for approval (the page says when access is granted). Fine-grained tokens also need
       'Read access to contents of all public gated repos'. Then re-run ./setup.sh." ;;
    esac
  fi

  echo "  downloading ${HF_REPO} (~${MODEL_DISK_GB} GB, resumable) -> ${MODEL_HOST_DIR}"
  download || die "download failed. Re-run ./setup.sh to resume; it keeps what is already on disk."
  [ -f "${MODEL_HOST_DIR}/config.json" ] || die "download finished but ${MODEL_HOST_DIR}/config.json is missing"
  # Every shard the index names must be on disk — huggingface_hub can exit 0
  # with half a model, which would otherwise fail minutes into vLLM's load.
  model_complete "${MODEL_HOST_DIR}" || die "download incomplete (weight shards missing). Re-run ./setup.sh to resume."
  touch "${MODEL_HOST_DIR}/.download-complete" 2>/dev/null \
    || docker run --rm -v "${MODEL_HOST_DIR}:/out" --entrypoint /bin/touch "${ENGINE_IMAGE}" /out/.download-complete
  echo "  model ok"
fi

# ---------------------------------------------------------------------------
step 5 "Preflight (8 gates)"
if [ -n "$(docker ps -q -f 'name=^momento-w[0-9]+$')" ]; then
  echo "  SKIPPED: worker containers are running and hold the GPUs."
  echo "  To re-check: ./stop.sh && ./setup.sh"
else
  "${KIT_DIR}/scripts/preflight.sh" "$@"
fi

# ---------------------------------------------------------------------------
step 6 "CacheBlend plugin"
if [ "$(cat "${PLUGIN_HOST_DIR}/.staged-from" 2>/dev/null)" = "${PLUGIN_IMAGE}" ]; then
  echo "  already staged from ${PLUGIN_IMAGE}"
else
  "${KIT_DIR}/scripts/stage-plugin.sh"
fi

echo
echo "SETUP COMPLETE.  Next:  ./start.sh $*"
