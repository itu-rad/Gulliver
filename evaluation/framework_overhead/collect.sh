#!/usr/bin/env bash
# Framework overhead: NoOp pipelines of depth 1-128 (bare and traced) and a
# payload sweep at depth 10 (bare).
#
#   usage: collect.sh <m3pro|gb10> [--runs N] [--dry-run] [--smoke]
source "$(dirname "$0")/../common/harness.sh"
harness_keep_awake "$0" "$@"
harness_init framework_overhead 11 "$@"
DEVICE=${HARGS[0]:-}
case $DEVICE in m3pro|gb10) ;; *) sed -n '2,5p' "$0"; exit 2 ;; esac

CFG=evaluation/framework_overhead/configs
DEPTHS="1 2 4 8 16 32 64 128"
SIZES="1024 1048576 10485760"
ARMS=(uninstrumented traced)

# Pin to one performance core on GB10; the Grace CPU mixes core types.
PINCMD=()
[ "$DEVICE" = gb10 ] && command -v taskset >/dev/null && PINCMD=(taskset -c "${PIN:-19}")

harness_outdir "$DEVICE"
# Local store: span export to a remote server would dominate these microbenchmarks.
harness_tracking local
export RADT_PRESENT=True RADT_TRACE_BACKEND=radt

run() {
  # usage: run <arm> <depth> <size> <r>
  local arm=$1 depth=$2 size=$3 r=$4 lab
  lab="noop_depth_${depth}_size_${size}_mode_ref_${arm}_${DEVICE}_r${r}"
  local env=(SUITE_DISABLE_TRACING=1)
  [ "$arm" = traced ] && env=(SUITE_PROC_TRACE=1)
  run_cell "$lab" env "${env[@]}" ${PINCMD[@]+"${PINCMD[@]}"} python main.py \
    "$CFG/noop_depth_${depth}_size_${size}_mode_ref_nolog.yml" -p 0 -e "$EXP_ID" \
    --label "$lab" ${MAX_QUERIES_ARGS[@]+"${MAX_QUERIES_ARGS[@]}"}
}

fail=0
log "depth sweep on $DEVICE, $RUNS run(s)"
for depth in $DEPTHS; do
  for r in $(seq 1 "$RUNS"); do
    # Alternate which arm goes first.
    off=$(( (r - 1) % 2 ))
    for i in 0 1; do
      run "${ARMS[$(( (off + i) % 2 ))]}" "$depth" 0 "$r" || fail=$((fail + 1))
    done
  done
done

log "payload sweep on $DEVICE, $RUNS run(s)"
for size in $SIZES; do
  for r in $(seq 1 "$RUNS"); do
    run uninstrumented 10 "$size" "$r" || fail=$((fail + 1))
  done
done

harness_done "$fail"
