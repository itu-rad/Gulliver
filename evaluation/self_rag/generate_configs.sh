#!/usr/bin/env bash
# Regenerate the configs in configs/ and configs_hw/ from the base configs
# (<task>_<arm>_<mlx|cuda>.yml). Writes to a temporary directory and refreshes
# only the files that are already present.
#
#   usage: generate_configs.sh [--out-dir DIR]   (default: refresh configs/ in place)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DEST=$HERE/configs
[ "${1:-}" = --out-dir ] && DEST=$2
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

ser() { python "$HERE/gen_serialized_configs.py" --out-dir "$TMP" "$@" >/dev/null; }
pip() { python "$HERE/gen_pipelined_configs.py" --out-dir "$TMP" --service '{"factoid/cuda":1,"factoid/mlx":1}' "$@" >/dev/null; }
rate() { python -c "print(1/$1)"; }

# Synchronous load: reasoning budgets, accuracy runs, 4-bit
ser --budget 0 --suffix serial0 --queries 12
ser --budget 256 --suffix serial256 --queries 12
ser --budget 1024 --suffix serial1024 --queries 12 --server
ser --budget 3072 --suffix serial3072 --queries 12
ser --budget 0 --suffix acc0 --queries 60
ser --budget 3072 --suffix acc3072 --queries 60
ser --suffix serial --precision q4 --queries 12 --server
ser --budget 3072 --suffix serial3072 --precision q4 --queries 12

# Poisson load. The 0.7x points are derived from the serial service times.
python "$HERE/gen_pipelined_configs.py" --out-dir "$TMP" --budget 1024 --suffix pipe1024q4m --precision q4 \
  --rho 0.7 --service '{"factoid/cuda":107,"factoid/mlx":66}' >/dev/null
python "$HERE/gen_pipelined_configs.py" --out-dir "$TMP" --budget 3072 --suffix pipe3072q4r153 --precision q4 \
  --rho 0.7 --service '{"factoid/cuda":107,"factoid/mlx":66}' >/dev/null
for s in 240 120 60 32; do
  pip --budget 1024 --suffix pipe1024r$s --rate "$(rate $s)"
  pip --budget 3072 --suffix pipe3072r$s --rate "$(rate $s)"
done
mv "$TMP/factoid_decomposed_shared_pipe1024r240_cuda.yml" "$TMP/factoid_decomposed_shared_pipe1024_cuda.yml"
mv "$TMP/factoid_decomposed_shared_server_pipe1024r240_cuda.yml" "$TMP/factoid_decomposed_shared_server_pipe1024_cuda.yml"
for s in 94 80 49 40 24; do
  pip --budget 1024 --suffix pipe1024q4r$s --precision q4 --rate "$(rate $s)"
  pip --budget 3072 --suffix pipe3072q4r$s --precision q4 --rate "$(rate $s)"
done
pip --budget 1024 --suffix pipe1024q4s16r24 --precision q4 --slots 16 --rate "$(rate 24)"
pip --budget 1024 --suffix pipe1024s32r8 --slots 32 --rate 0.125
pip --budget 3072 --suffix pipe3072s32r8 --slots 32 --rate 0.125

n=0
for f in "$TMP"/*.yml; do
  b=$(basename "$f")
  if [ "$DEST" != "$HERE/configs" ] || [ -f "$HERE/configs/$b" ]; then
    mkdir -p "$DEST"; cp "$f" "$DEST/$b"; n=$((n + 1))
  fi
done
python "$HERE/gen_hw_configs.py" ${1:+--out-dir "$DEST/../configs_hw"} >/dev/null
echo "refreshed $n configs in $DEST"
