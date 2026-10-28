#!/usr/bin/env bash
# Self-RAG case study: decomposed and monolithic pipelines with in-process and
# engine (vLLM) inference.
#
#   usage: collect.sh <m3pro|gb10> --pass <serial|acc|load|hw> [--runs N] [--dry-run] [--smoke]
#
#   serial  synchronous load across reasoning budgets, 16- and 4-bit
#   acc     60-question accuracy runs at each budget (m3pro)
#   load    Poisson load at 0.7x/1.35x/2.7x capacity, and the saturation runs
#   hw      GPU metrics (Nsight) for the serial cells (gb10)
#
# Engine cells need vLLM (gb10) or vllm-metal (m3pro); set VLLM_EXECUTABLE to
# its `vllm` binary if it is not on PATH.
source "$(dirname "$0")/../common/harness.sh"
harness_keep_awake "$0" "$@"
harness_init self_rag "" "$@"
DEVICE=""; PASS=""
set -- ${HARGS[@]+"${HARGS[@]}"}
while [ $# -gt 0 ]; do
  case $1 in --pass) PASS=$2; shift ;; *) DEVICE=$1 ;; esac
  shift
done
case $DEVICE in m3pro) ENGINE=mlx ;; gb10) ENGINE=cuda ;; *) sed -n '2,14p' "$0"; exit 2 ;; esac

# Each entry: "<runs> <config stem>...". Stems omit the _mlx/_cuda suffix.
CELLS=()
case $DEVICE/$PASS in
  m3pro/serial) CELLS=(
    "5 factoid_decomposed_shared_serial factoid_monolith_4b_serial"
    "2 factoid_decomposed_shared_serial0 factoid_monolith_4b_serial0"
    "2 factoid_decomposed_shared_serial256 factoid_monolith_4b_serial256"
    "2 factoid_decomposed_shared_serial1024 factoid_monolith_4b_serial1024"
    "1 factoid_monolith_4b_serial3072"
    "1 factoid_decomposed_shared_serialq4 factoid_decomposed_shared_server_serialq4"
    "1 factoid_monolith_4b_serial3072q4") ;;
  gb10/serial) CELLS=(
    "5 factoid_decomposed_shared_serial factoid_monolith_4b_serial"
    "1 factoid_decomposed_shared_serial0 factoid_monolith_4b_acc0"
    "1 factoid_decomposed_shared_serial256 factoid_monolith_4b_serial256"
    "1 factoid_decomposed_shared_serial1024 factoid_monolith_4b_serial1024 factoid_decomposed_shared_server_serial1024"
    "1 factoid_monolith_4b_serial3072"
    "1 factoid_decomposed_shared_serialq4 factoid_decomposed_shared_server_serialq4"
    "1 factoid_monolith_4b_serial3072q4") ;;
  m3pro/acc) CELLS=(
    "1 factoid_decomposed_shared_acc factoid_monolith_4b_acc multihop_decomposed_shared_acc multihop_monolith_4b_acc"
    "1 factoid_decomposed_shared_acc0 factoid_monolith_4b_acc0 multihop_decomposed_shared_acc0 multihop_monolith_4b_acc0"
    "1 factoid_monolith_4b_acc3072 multihop_monolith_4b_acc3072") ;;
  m3pro/load) CELLS=(
    "1 factoid_decomposed_shared_pipe1024q4m factoid_decomposed_shared_server_pipe1024q4m factoid_monolith_4b_server_pipe3072q4r94"
    "1 factoid_decomposed_shared_pipe1024q4r49 factoid_decomposed_shared_server_pipe1024q4r49 factoid_monolith_4b_server_pipe3072q4r49"
    "1 factoid_decomposed_shared_pipe1024q4r24 factoid_decomposed_shared_server_pipe1024q4r24 factoid_monolith_4b_server_pipe3072q4r24"
    "1 factoid_decomposed_shared_server_pipe1024q4s16r24") ;;
  gb10/load) CELLS=(
    "1 factoid_decomposed_shared_pipe1024 factoid_decomposed_shared_server_pipe1024 factoid_monolith_4b_server_pipe3072r240"
    "1 factoid_decomposed_shared_pipe1024r120 factoid_decomposed_shared_server_pipe1024r120 factoid_monolith_4b_server_pipe3072r120"
    "1 factoid_decomposed_shared_pipe1024r60 factoid_decomposed_shared_server_pipe1024r60 factoid_monolith_4b_server_pipe3072r60"
    "1 factoid_decomposed_shared_pipe1024q4m factoid_decomposed_shared_server_pipe1024q4m factoid_monolith_4b_server_pipe3072q4r153"
    "1 factoid_decomposed_shared_pipe1024q4r80 factoid_decomposed_shared_server_pipe1024q4r80 factoid_monolith_4b_server_pipe3072q4r80"
    "1 factoid_decomposed_shared_pipe1024q4r40 factoid_decomposed_shared_server_pipe1024q4r40 factoid_monolith_4b_server_pipe3072q4r40"
    "1 factoid_decomposed_shared_server_pipe1024s32r8 factoid_monolith_4b_server_pipe3072s32r8") ;;
  gb10/hw) CELLS=(
    "1 decomposed_serial0 decomposed_serial1024 decomposed_serial1024nf4 decomposed_engine_serial1024 decomposed_engine_serial1024mxfp4") ;;
  *) sed -n '2,14p' "$0"; exit 2 ;;
esac

harness_outdir "$DEVICE"
harness_tracking
export RADT_PRESENT=True RADT_TRACE_BACKEND=radt SUITE_PROC_TRACE=1
export RADT_TRACE_BATCH_EVENTS=20000 RADT_TRACE_BATCH_SECONDS=30
[ "$(uname)" = Darwin ] && export PATH="/opt/homebrew/bin:$PATH"
if [ "$PASS" = hw ]; then
  export RADT_GPUMETRICS_DIR="$OUT/gpumetrics"
  [ "$DRY_RUN" = 0 ] && mkdir -p "$RADT_GPUMETRICS_DIR"
fi

config_for() {
  if [ "$PASS" = hw ]; then echo "evaluation/self_rag/configs_hw/$1.yml"
  else echo "evaluation/self_rag/configs/$1_$ENGINE.yml"; fi
}

run() {
  local stem=$1 r=$2 cfg lab="$1_${DEVICE}_r$2" listeners n0=""
  cfg=$(config_for "$stem")
  listeners=$(python -c "import sys,yaml;print(' '.join('RADT_LISTENER_'+l.upper()+'=True' for l in yaml.safe_load(open(sys.argv[1])).get('listeners') or []))" "$cfg")
  [ "$PASS" = hw ] && [ "$DRY_RUN" = 0 ] && [ ! -f "$OUT/$lab.done" ] && n0=$(ls "$RADT_GPUMETRICS_DIR" | wc -l)
  run_cell "$lab" env $listeners python main.py "$cfg" -p 0 -e "$EXP_ID" --label "$lab" \
    ${MAX_QUERIES_ARGS[@]+"${MAX_QUERIES_ARGS[@]}"} || return 1
  if [ -n "$n0" ] && [ "$(ls "$RADT_GPUMETRICS_DIR" | wc -l)" -le "$n0" ]; then
    log "  $lab has no GPU metrics (see the gpumetrics lines in the log)"
    rm -f "$OUT/$lab.done"; return 1
  fi
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
      run "$stem" "$r" || fail=$((fail + 1))
    done
  done
done
harness_done "$fail"
