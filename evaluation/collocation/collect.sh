#!/usr/bin/env bash
# Collocation on GB10: a decomposed Self-RAG foreground on vLLM beside a CLIP
# image encoder or an e5-base indexer, under MPS and time-slicing.
#
#   usage: collect.sh gb10 --pass <alone|colloc|capture> [--runs N] [--dry-run] [--smoke]
#
#   alone    the two background jobs alone
#   colloc   foreground alone and collocated, Qwen3.5 4B/9B/27B
#   capture  10 kHz Nsight capture of the 4B foreground beside each background
#
# Needs nsys, nvidia-smi, the CUDA MPS daemon, and vLLM (VLLM_EXECUTABLE).
source "$(dirname "$0")/../common/harness.sh"
harness_init collocation "" "$@"
DEVICE=""; PASS=""
set -- ${HARGS[@]+"${HARGS[@]}"}
while [ $# -gt 0 ]; do
  case $1 in --pass) PASS=$2; shift ;; *) DEVICE=$1 ;; esac
  shift
done
[ "$DEVICE" = gb10 ] || { sed -n '2,11p' "$0"; exit 2; }

# Each entry: "<runs> <config stem>...".
HZ=""
case $PASS in
  alone) CELLS=("1 fp_serial_encode_1 fp_serial_index_e5base_32") ;;
  colloc) CELLS=(
    "3 e5s_4b_baseline e5s_4b_bg_encode_mps e5s_4b_bg_encode_ts"
    "1 e5s_9b_baseline e5s_9b_bg_encode_mps e5s_9b_bg_encode_ts"
    "1 e5s_27b_baseline e5s_27b_bg_encode_mps e5s_27b_bg_encode_ts"
    "1 e5b_4b_baseline e5b_4b_bg_index_mps e5b_4b_bg_index_ts"
    "1 e5b_9b_baseline e5b_9b_bg_index_mps e5b_9b_bg_index_ts"
    "1 e5b_27b_baseline e5b_27b_bg_index_mps e5b_27b_bg_index_ts") ;;
  capture) HZ=10000; CELLS=("1 e5r_4b_bg_encodes_mps e5r_4b_bg_encodes_ts e5q_4b_bg_index_mps e5q_4b_bg_index_ts") ;;
  *) sed -n '2,11p' "$0"; exit 2 ;;
esac

harness_outdir "$DEVICE"
harness_tracking
export RADT_PRESENT=True RADT_TRACE_BACKEND=radt SUITE_PROC_TRACE=1
export RADT_TRACE_BATCH_EVENTS=20000 RADT_TRACE_BATCH_SECONDS=30
export RADT_LISTENER_DCGMI=True RADT_LISTENER_TOP=True RADT_LISTENER_VLLM=True
export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps-$USER}
export CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-mps-log-$USER}
[ "$DRY_RUN" = 0 ] && mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
SAMPLERS=(python evaluation/common/gb10_samplers.py all --dir "$OUT"
          --config evaluation/common/nsys_gb10_occ.config ${HZ:+--hz $HZ})

wait_idle() {
  # The previous cell's processes can take a while to leave the GPU.
  for _ in $(seq 1 60); do
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && return 0
    sleep 10
  done
  log "GPU still busy after 10 min"; return 1
}

run() {
  local stem=$1 r=$2 lab="$1_r$2" cfg="evaluation/collocation/configs/$1_cuda.yml" rc secs rows
  local cmd=(python main.py "$cfg" -e "$EXP_ID" --label "$lab" ${MAX_QUERIES_ARGS[@]+"${MAX_QUERIES_ARGS[@]}"})
  if [ "$DRY_RUN" = 1 ]; then
    echo "${SAMPLERS[*]} --label $lab &"; echo "${cmd[*]}"; return 0
  fi
  [ -f "$OUT/$lab.done" ] && { log "  [skip] $lab (done)"; return 0; }
  rm -f "$OUT/${lab}_"*
  wait_idle || return 1
  "${SAMPLERS[@]}" --label "$lab" > "$OUT/logs/${lab}_samplers.log" 2>&1 &
  local sampid=$! start; sleep 3; start=$(date +%s)
  "${cmd[@]}" > "$OUT/logs/${lab}_main.log" 2>&1; rc=$?
  secs=$(( $(date +%s) - start ))
  kill -TERM "$sampid"; wait "$sampid"
  rows=$(cat "$OUT/${lab}_"{self,bg}*.csv 2>/dev/null | wc -l | tr -d ' ')
  printf '%s\t%s\t%s\t%s\t\n' "$lab" "$rc" "$secs" "$rows" >> "$SUM"
  log "  $lab rc=$rc ${secs}s rows=$rows"
  [ "$rc" = 0 ] && [ "$rows" -gt 0 ] || { log "  $lab failed (see logs/${lab}_main.log)"; return 1; }
  [ -s "$OUT/${lab}_nsys.csv" ] || { log "  $lab has no Nsight samples (see logs/${lab}_samplers.log)"; return 1; }
  touch "$OUT/$lab.done"
}

fail=0
for entry in "${CELLS[@]}"; do
  read -r runs stems <<<"$entry"
  read -ra stems <<<"$stems"
  [ -n "$RUNS" ] && runs=$RUNS
  n=${#stems[@]}
  for r in $(seq 1 "$runs"); do
    # Rotate the order of the cells in each repetition.
    off=$(( (r - 1) % n ))
    for i in $(seq 0 $(( n - 1 ))); do
      stem=${stems[$(( (off + i) % n ))]}
      # A smoke covers the 4B cells and one MPS cell per larger model.
      if [ "$SMOKE" = 1 ]; then
        case $stem in *_4b_*|fp_*|e5s_9b_bg_encode_mps|e5s_27b_bg_encode_mps) ;; *) continue ;; esac
      fi
      run "$stem" "$r" || fail=$((fail + 1))
    done
  done
done
harness_done "$fail"
