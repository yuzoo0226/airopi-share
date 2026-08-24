#!/usr/bin/env bash
# =============================================================================
#  Download an openpi HSR checkpoint from Hugging Face.
#
#  Plain curl is used instead of huggingface_hub / git-lfs so the script has no
#  Python dependency beyond the standard library, and files are fetched in
#  parallel because a single connection to huggingface.co is the bottleneck.
#
#  Usage: scripts/ros2/download_base_model.sh [repo_id] [dest_dir] [jobs]
#  Default: airoa-org/airoa-pi05-hsr-base -> <parent of this repo>/checkpoints/
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"

REPO_ID="${1:-airoa-org/airoa-pi05-hsr-base}"
DEST="${2:-$(cd "${REPO_ROOT}/.." && pwd)/checkpoints/$(basename "${REPO_ID}")}"
JOBS="${3:-10}"
BASE="https://huggingface.co/${REPO_ID}/resolve/main"

mkdir -p "${DEST}"

echo "[INFO] listing ${REPO_ID} ..."
curl -sfL "https://huggingface.co/api/models/${REPO_ID}" \
  | python3 -c 'import json,sys; [print(s["rfilename"]) for s in json.load(sys.stdin)["siblings"]]' \
  > "${DEST}/.filelist"

TOTAL=$(wc -l < "${DEST}/.filelist")
echo "[INFO] ${TOTAL} files -> ${DEST} (${JOBS} parallel downloads)"

export DEST BASE
fetch_one() {
  local f="$1" out="${DEST}/$1"
  mkdir -p "$(dirname "${out}")"
  [ -s "${out}" ] && return 0
  curl -sfL --retry 8 --retry-delay 3 --retry-all-errors -o "${out}.part" "${BASE}/${f}" \
    && mv "${out}.part" "${out}" \
    && echo "  ok ${f}"
}
export -f fetch_one

xargs -a "${DEST}/.filelist" -I{} -P "${JOBS}" bash -c 'fetch_one "$@"' _ {}

MISSING=0
while IFS= read -r f; do
  [ -s "${DEST}/${f}" ] || { echo "[WARN] missing: ${f}"; MISSING=$((MISSING + 1)); }
done < "${DEST}/.filelist"

if [ "${MISSING}" -gt 0 ]; then
  echo "[ERROR] ${MISSING} file(s) missing - re-run to resume." >&2
  exit 1
fi

echo "[INFO] done: $(du -sh "${DEST}" | cut -f1) in ${DEST}"
echo "[INFO] point the policy server at it with"
echo "         POLICY_CHECKPOINT_DIR=/home/openpi/checkpoints/$(basename "${DEST}")"
