#!/usr/bin/env bash
# MLPerf 3D-UNet (KiTS19) as a two-stage pipeline, and the MLPerf reference
# implementation on the same device.
#
#   usage: collect.sh <m3pro|gb10> --pass <perf|acc|reference> [--runs N] [--dry-run] [--smoke]
#
#   perf       latency of preprocessing and inference over the 42 cases (default 6 runs)
#   acc        Dice score of the 42 cases (1 run)
#   reference  MLPerf reference harness, SingleStream performance and accuracy
source "$(dirname "$0")/../common/harness.sh"
harness_keep_awake "$0" "$@"
harness_init unet3d 6 "$@"
DEVICE=""; PASS=""
set -- ${HARGS[@]+"${HARGS[@]}"}
while [ $# -gt 0 ]; do
  case $1 in --pass) PASS=$2; shift ;; *) DEVICE=$1 ;; esac
  shift
done
case "$DEVICE/$PASS" in m3pro/perf|m3pro/acc|gb10/perf|gb10/acc|gb10/reference) ;; *) sed -n '2,11p' "$0"; exit 2 ;; esac
[ "$PASS" = acc ] && RUNS=1

harness_outdir "$DEVICE"

if [ "$PASS" = reference ]; then
  REF=${MLPERF_INFERENCE_DIR:-$ROOT/tmp/mlperf-inference}
  BENCH=$REF/vision/medical_imaging/3d-unet-kits19
  DATA=$ROOT/data/kits19
  MODEL=$ROOT/models/3dunet_kits19/3dunet_kits19_pytorch.ptc
  CONF=$ROOT/evaluation/unet3d/mlperf_reference/user.conf
  if [ "$SMOKE" = 1 ]; then
    CONF=$OUT/user_smoke.conf
    [ "$DRY_RUN" = 0 ] && printf '*.SingleStream.target_latency = 8000\n*.SingleStream.min_duration = 0\n*.SingleStream.min_query_count = 2\n*.SingleStream.max_query_count = 2\n' > "$CONF"
  fi
  steps=(
    "[ -d $BENCH ] || { git clone --filter=blob:none --sparse https://github.com/mlcommons/inference.git $REF && git -C $REF sparse-checkout set vision/medical_imaging/3d-unet-kits19 && git -C $REF checkout 3695ef0 && git -C $REF apply $ROOT/evaluation/unet3d/mlperf_reference/mlperf_reference.patch; }"
    "[ -f $DATA/preprocessed_mlperf/preprocessed_files.pkl ] || (cd $BENCH && python preprocess.py --raw_data_dir $DATA/raw --results_dir $DATA/preprocessed_mlperf --mode preprocess)"
    "mkdir -p $OUT/reference/perf && cd $BENCH && python run.py --backend=pytorch --scenario=SingleStream --model=$MODEL --preprocessed_data_dir=$DATA/preprocessed_mlperf --user_conf=$CONF && cp build/logs/mlperf_log_* $OUT/reference/perf/"
  )
  [ "$SMOKE" = 0 ] && steps+=("mkdir -p $OUT/reference/acc && cd $BENCH && python run.py --backend=pytorch --scenario=SingleStream --model=$MODEL --preprocessed_data_dir=$DATA/preprocessed_mlperf --user_conf=$CONF --accuracy --postprocessed_data_dir=$OUT/reference/acc/postprocessed && cp build/logs/mlperf_log_* $OUT/reference/acc/")
  for s in "${steps[@]}"; do
    if [ "$DRY_RUN" = 1 ]; then echo "$s"; continue; fi
    log "$s"
    (eval "$s") || { log "failed"; exit 1; }
  done
  [ "$DRY_RUN" = 0 ] && grep -h 'Result is\|90.00 percentile' "$OUT"/reference/perf/mlperf_log_summary.txt | tee -a "$LOG"
  exit 0
fi

harness_tracking
export RADT_PRESENT=True RADT_TRACE_BACKEND=radt SUITE_PROC_TRACE=1

fail=0
log "$PASS on $DEVICE, $RUNS run(s)"
for r in $(seq 1 "$RUNS"); do
  lab="unet3d_42_${PASS}_${DEVICE}_r${r}"
  run_cell "$lab" python main.py "evaluation/unet3d/configs/unet3d_42_${PASS}_${DEVICE}.yml" \
    -p 0 -e "$EXP_ID" --label "$lab" ${MAX_QUERIES_ARGS[@]+"${MAX_QUERIES_ARGS[@]}"} || fail=$((fail + 1))
done
if [ "$PASS" = acc ] && [ "$DRY_RUN" = 0 ]; then
  [ -s "$OUT/dice.csv" ] && log "Dice rows: $(( $(wc -l < "$OUT/dice.csv") - 1 ))" || { log "no dice.csv"; fail=$((fail + 1)); }
fi
harness_done "$fail"
