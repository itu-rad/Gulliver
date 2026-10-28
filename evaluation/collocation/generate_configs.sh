#!/usr/bin/env bash
# Regenerate configs/ (collocation cells and the backgrounds alone).
#   usage: generate_configs.sh [--out-dir DIR]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
gen() { python "$HERE/generate_configs.py" "$@"; }
SERVICE='{"4b":129,"9b":248.6,"27b":707.9}'
gen real --tag e5s --sizes 4b,9b,27b --service "$SERVICE" --cells baseline,bg_encode_mps,bg_encode_ts "$@"
gen real --tag e5b --sizes 4b,9b,27b --service "$SERVICE" --cells baseline,bg_index_mps,bg_index_ts "$@"
gen real --tag e5r --sizes 4b --service "$SERVICE" --queries 4 --cells bg_encodes_mps,bg_encodes_ts "$@"
gen real --tag e5q --sizes 4b --service "$SERVICE" --queries 4 --cells bg_index_mps,bg_index_ts "$@"
gen alone --tag e5s --sizes 4b --cells bg_encode_mps,bg_index_mps "$@"
