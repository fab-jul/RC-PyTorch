#!/bin/bash

set -e

SRC_DIR="$(dirname "$0")"
FIGS_DIR="$SRC_DIR/../figs"
OVERVIEW_PNG="$FIGS_DIR/overview.png"

if [[ ! -f "$OVERVIEW_PNG" ]]; then
    echo "ERROR: Expected a PNG at $OVERVIEW_PNG, nothing found!"
    exit 1
fi

OUT_BPG=overview.tmp.bpg
OUT_PNG=overview_recon.png
bpgenc -o $OUT_BPG "$OVERVIEW_PNG"
bpgdec -o $OUT_PNG $OUT_BPG

if [[ -f $OUT_PNG ]]; then
    echo "SUCCESS: bpgenc, bpgdec available"
else
    echo "ERROR: Expected output file not found"
fi

rm $OUT_BPG
rm $OUT_PNG

