#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# scripts/pyenv.sh — the host-side Python environment, created the same way on
# Ubuntu 22.04 / 24.04, Amazon Linux 2023, RHEL / Rocky and Debian.
#
#   scripts/pyenv.sh            # creates .venv in the kit (once; re-run to update)
#   source .venv/bin/activate   # optional: then `python -m swarm.simulate ...`
#
# Every kit script uses .venv/bin/python when it exists, so activating is only
# for running the swarm tools by hand.
#
# Why uv and not `pip install -r requirements.txt`: Ubuntu 24.04+ refuses pip
# outside a virtualenv (PEP 668), Ubuntu does not ship the venv module by
# default (python3-venv), and Amazon Linux 2023's python3 is 3.9. uv is one
# static binary (installed into ~/.local/bin if missing, no root, needs curl and
# tar); it uses the system Python 3.12 if there is one and otherwise downloads a
# standalone CPython 3.12, so every host runs the same interpreter.
#
# Offline or no access to astral.sh / github.com: install uv some other way
# (pipx install uv, or copy the binary), or set PYENV_NO_UV=1 to fall back to
# `python3 -m venv` (needs a system Python >= 3.10 with its venv module).
# ---------------------------------------------------------------------------
set -Eeuo pipefail
KIT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV="${KIT_DIR}/.venv"
REQ="${KIT_DIR}/requirements.txt"
PY_SPEC=${PY_SPEC:-3.12}    # the system 3.12 if there is one, else a standalone CPython 3.12

if [ "${PYENV_NO_UV:-0}" != 1 ]; then
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
  if ! command -v uv >/dev/null 2>&1; then
    command -v curl >/dev/null 2>&1 || { echo "FATAL: curl is needed to install uv (or install uv yourself, or PYENV_NO_UV=1)" >&2; exit 1; }
    echo "installing uv into ~/.local/bin ..."
    curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh >/dev/null \
      || { echo "FATAL: uv install failed (no access to astral.sh?). Install uv another way, or PYENV_NO_UV=1" >&2; exit 1; }
  fi
  echo "creating ${VENV} (Python ${PY_SPEC}) with $(uv --version)"
  uv venv --quiet --allow-existing --python "${PY_SPEC}" "${VENV}"
  uv pip install --quiet --python "${VENV}/bin/python" -r "${REQ}"
else
  PY3=$(command -v python3 || true)
  [ -n "${PY3}" ] || { echo "FATAL: no python3 on PATH" >&2; exit 1; }
  "${PY3}" -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
    || { echo "FATAL: $(${PY3} --version) is older than 3.10: use uv (unset PYENV_NO_UV)" >&2; exit 1; }
  "${PY3}" -m venv "${VENV}" \
    || { echo "FATAL: python3 -m venv failed (on Ubuntu/Debian: sudo apt install python3-venv)" >&2; exit 1; }
  "${VENV}/bin/python" -m pip install -q --upgrade pip
  "${VENV}/bin/python" -m pip install -q -r "${REQ}"
fi

"${VENV}/bin/python" -c 'import numpy, pyarrow, openai, transformers, tiktoken' \
  || { echo "FATAL: ${VENV} is missing packages from requirements.txt" >&2; exit 1; }
echo "ok  ${VENV}: $("${VENV}/bin/python" --version)"
