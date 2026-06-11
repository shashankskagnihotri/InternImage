#!/usr/bin/env bash
set -euo pipefail

ROOT="/ceph/sagnihot/datasets/cityscapes"
mkdir -p "$ROOT"
cd "$ROOT"

# Export these before running:
: "${CITYSCAPES_USER:?Please export CITYSCAPES_USER}"
: "${CITYSCAPES_PASS:?Please export CITYSCAPES_PASS}"

# cityscapesScripts expects these env vars (the downloader uses them)
export CITYSCAPES_USERNAME="$CITYSCAPES_USER"
export CITYSCAPES_PASSWORD="$CITYSCAPES_PASS"

# Download exactly the file you want
csDownload leftImg8bit_trainextra.zip

echo "Done: $ROOT/leftImg8bit_trainextra.zip"
