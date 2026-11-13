#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# scripts/stage-plugin.sh — get the cacheblend-plugin payload onto the host.
#
# In the Helm path the mutating webhook runs the payload image as an init
# container and copies its contents into a shared volume the vLLM container
# mounts at /cbplugin. There is no webhook on bare metal, so we do that copy
# once, by hand, and bind-mount the result.
# ---------------------------------------------------------------------------
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=../lib/env.sh
source "${ROOT}/lib/env.sh"; load_env || exit 1

: "${PLUGIN_IMAGE:?}" "${PLUGIN_HOST_DIR:?}" "${TM_API_KEY:?}"

registry_login || { echo "FATAL: ${REGISTRY} login failed" >&2; exit 1; }
docker pull "${PLUGIN_IMAGE}"

case "${PLUGIN_HOST_DIR}" in
  */cbplugin) ;;
  *) echo "FATAL: PLUGIN_HOST_DIR=${PLUGIN_HOST_DIR} must end in /cbplugin — refusing to rm -rf it" >&2; exit 1 ;;
esac
rm -rf "${PLUGIN_HOST_DIR}"
mkdir -p "${PLUGIN_HOST_DIR}"

CID=$(docker create "${PLUGIN_IMAGE}")
trap 'docker rm -f "$CID" >/dev/null 2>&1 || true' EXIT

# The payload's internal path is not contractual across releases, so find the
# lmcache_cacheblend package rather than assuming a location.
STAGE=$(mktemp -d)
docker export "$CID" | tar -x -C "$STAGE" 2>/dev/null || true

PKG=$(find "$STAGE" -maxdepth 6 -type d -name 'lmcache_cacheblend' -print -quit)
if [ -z "$PKG" ]; then
  echo "Could not locate lmcache_cacheblend in ${PLUGIN_IMAGE}." >&2
  echo "Top-level layout of the payload image:" >&2
  ls -la "$STAGE" >&2
  echo "Inspect it and set PLUGIN_HOST_DIR to the parent of lmcache_cacheblend by hand." >&2
  exit 1
fi

# PYTHONPATH must point at the PARENT of the package directory.
cp -a "$(dirname "$PKG")/." "${PLUGIN_HOST_DIR}/"
rm -rf "$STAGE"

echo "Plugin staged at ${PLUGIN_HOST_DIR}:"
ls -la "${PLUGIN_HOST_DIR}"
[ -d "${PLUGIN_HOST_DIR}/lmcache_cacheblend" ] || { echo "FAIL: package missing after copy" >&2; exit 1; }
echo "${PLUGIN_IMAGE}" > "${PLUGIN_HOST_DIR}/.staged-from"   # setup.sh skips re-staging on a match
echo "OK"
