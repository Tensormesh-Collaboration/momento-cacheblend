#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# scripts/verify.sh --worker <i> — prove the things that matter, for one worker:
#   A. the LMCache server is on CUDA, not the CPU stub
#   B. register_kv_caches cleared (the 300 s hang did not happen)
#   C. vLLM is serving
#   D. L2 is really written (Valkey keys under our prefix, or fs files)
#   E. with CacheBlend, the blend module is loaded and blending fires
#      (scripts/smoke_cb.py: the same body behind a new preamble is reused)
# ---------------------------------------------------------------------------
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=../lib/env.sh
source "${ROOT}/lib/env.sh"; load_env || exit 1

W=${WORKER_BASE:-0}
[ "${1:-}" = --worker ] && W=${2:?--worker needs an index}
L=$(worker_log_dir "$W")
PORT=$(worker_port "$W")
HTTP=$(worker_http_port "$W")
pass=0; fail=0
chk() { if eval "$2"; then echo "  PASS  $1"; pass=$((pass+1)); else echo "  FAIL  $1"; fail=$((fail+1)); fi; }

echo "== worker ${W}: A. LMCache backend =="
chk "not on StubCPUDevice"        "! grep -qi 'StubCPUDevice' '$L/lmcache-server.log'"
chk "no 'driver too old' warning" "! grep -qi 'driver on your system is too old' '$L/lmcache-server.log'"

echo "== B. KV cache registration =="
chk "no register_kv_caches timeout"     "! grep -q 'did not respond to register_kv_caches' '$L/vllm.log'"
chk "no cudaErrorMapBufferObjectFailed" "! grep -qi 'mapping of buffer object failed' '$L/lmcache-server.log' '$L/vllm.log'"
chk "KV cache registered"               "grep -qi 'Registered KV cache' '$L/lmcache-server.log' '$L/vllm.log'"

echo "== C. serving =="
chk "OpenAI endpoint up on :${PORT}" "curl -sf http://localhost:${PORT}/v1/models >/dev/null"

if [ "${CACHE_MODE}" = blend ]; then
  echo "== E. CacheBlend module =="
  # Only the blend module reports active_cb_lookups in /status (see LMCache
  # engine_bench/config.py: server_runs_blend_module).
  chk "blend module loaded (/status has active_cb_lookups)" \
      "curl -sf http://127.0.0.1:${HTTP}/status | grep -q active_cb_lookups"
  chk "CacheBlend reuses a shifted body (non-prefix hits > 0)" \
      "python3 '${KIT_DIR}/scripts/smoke_cb.py' --port ${PORT} --metrics http://127.0.0.1:${HTTP}/metrics"
  grep -h 'CB: auto-selected\|CB: preferring kv cache block size\|CB: turned off' "$L/vllm.log" 2>/dev/null | tail -3 || true
fi

count_l2() {
  case "${L2_BACKEND}" in
    valkey) python3 "${KIT_DIR}/scripts/valkey.py" count 2>/dev/null || echo 0 ;;
    fs) find "${L2_HOST_PATH}" -type f 2>/dev/null | wc -l ;;
  esac
}

if [ "${L2_BACKEND}" != none ]; then
  echo "== D. L2 (${L2_BACKEND}) =="
  before=$(count_l2)
  VLLM_PORT=$PORT python3 - <<'PY'
import json, os, urllib.request
port = os.environ["VLLM_PORT"]
body = json.dumps({
    "model": os.environ.get("SERVED_NAME", ""),
    "messages": [{"role": "user", "content": "Summarize this. " + ("The quick brown fox jumps over the lazy dog. " * 1500)}],
    "max_tokens": 8,
}).encode()
req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", data=body,
                             headers={"Content-Type": "application/json"})
urllib.request.urlopen(req, timeout=600).read()
PY
  sleep 10
  after=$(count_l2)
  echo "  L2 objects: ${before:-0} -> ${after:-0}"
  chk "L2 received new objects" "[ \"${after:-0}\" -gt \"${before:-0}\" ]"
fi

echo "--------------------------------------------"
echo "worker ${W}: PASS ${pass}   FAIL ${fail}"
[ "$fail" -eq 0 ] || exit 1
