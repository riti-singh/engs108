#!/usr/bin/env bash
# Download the pooled hidden states used by the exp02 linear probes and verify their checksums.
# Usage: exp02/fetch_pooled_hidden.sh [dest_dir]   (needs Kaggle CLI auth with access to the private dataset)
set -euo pipefail
DEST="${1:-$(dirname "$0")/hidden}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$DEST"
kaggle datasets download ritisingh123/engs108-exp02-pooled-hidden-states -p "$DEST" --unzip
(cd "$DEST" && sha256sum -c "$HERE/artifacts/pooled_hidden_SHA256SUMS")
echo "OK. Run: EXP02_HIDDEN_DIR=$DEST python exp02_diagnostics_stylometric.py"
