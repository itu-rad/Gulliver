#!/usr/bin/env bash
# Cost of decomposition: EfficientNetV2 fine-tuning as a two-stage pipeline
# (untraced, per-query rows on) against the hand-written loop in
# baseline_finetune.py, on identical batches in identical order.
#
#   usage: collect.sh <m3pro|gb10> [--runs N] [--dry-run] [--smoke]
#
# On Linux the CPU latency QoS is held at 0 us for the whole collection, which
# needs write access to /dev/cpu_dma_latency (sudo chmod 666 /dev/cpu_dma_latency).
source "$(dirname "$0")/../common/harness.sh"
harness_keep_awake "$0" "$@"
if [ "$(uname)" = Linux ] && [ -z "${_QOS_HELD:-}" ] && [[ " $* " != *" --dry-run "* ]]; then
  [ -w /dev/cpu_dma_latency ] || { echo "/dev/cpu_dma_latency is not writable" >&2; exit 3; }
  export _QOS_HELD=1
  exec python "$(dirname "$0")/../common/cpu_latency_qos.py" --require --max-us 0 -- "$0" "$@"
fi
harness_init modularity_overhead 11 "$@"
DEVICE=${HARGS[0]:-}
case $DEVICE in m3pro|gb10) ;; *) sed -n '2,9p' "$0"; exit 2 ;; esac

# Pin to the performance cores on GB10.
PINCMD=()
[ "$DEVICE" = gb10 ] && command -v taskset >/dev/null && PINCMD=(taskset -c "${PIN:-5-9,15-19}")

harness_outdir "$DEVICE"
harness_tracking local
export RADT_PRESENT=True RADT_TRACE_BACKEND=radt

ARMS=(monolith pipeline)

run() {
  # usage: run <arm> <cell> <cfg> <r>; arms of one repetition share the data seed.
  local arm=$1 cell=$2 cfg=$3 r=$4 lab steps
  lab="mod_${cell}_${arm}_${DEVICE}_r${r}"
  steps=$MAXB; [ "$SMOKE" = 1 ] && steps=2
  if [ "$arm" = monolith ]; then
    run_cell "$lab" ${PINCMD[@]+"${PINCMD[@]}"} python evaluation/modularity_overhead/baseline_finetune.py \
      --device "$TORCH_DEVICE" --model "$MODEL" --weights "$WEIGHTS" \
      --batch-size "$BATCH" --num-workers 0 --max-batches "$steps" \
      --label "$lab" --no-radt --seed "$r"
  else
    run_cell "$lab" env SUITE_DISABLE_TRACING=1 SUITE_DATA_SEED="$r" ${PINCMD[@]+"${PINCMD[@]}"} \
      python main.py "$cfg" -p 0 -e "$EXP_ID" --label "$lab" --pipeline-logs on \
      ${MAX_QUERIES_ARGS[@]+"${MAX_QUERIES_ARGS[@]}"}
  fi
}

fail=0
for cfg in evaluation/modularity_overhead/configs/generated/mod_*_"$DEVICE".yml; do
  cell=$(basename "$cfg" .yml); cell=${cell#mod_}; cell=${cell%_"$DEVICE"}
  # Model, weights, batch size, step count and device come from the config, so
  # both arms run the same workload.
  read -r MODEL WEIGHTS BATCH MAXB TORCH_DEVICE <<<"$(python - "$cfg" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))["pipelines"][0]
dl = next(s for s in c["stages"] if "DataLoader" in s["component"])["config"]
tr = next(s for s in c["stages"] if "Classification" in s["component"])["config"]
print(tr["model"]["component"].rsplit(".", 1)[-1], dl["dataset"]["weights"],
      dl["batch_size"], c["loadgen"]["max_queries"], tr["device"])
PY
)"
  log "$cell on $DEVICE, $RUNS run(s)"
  for r in $(seq 1 "$RUNS"); do
    # Alternate which arm goes first.
    off=$(( (r - 1) % 2 ))
    for i in 0 1; do
      run "${ARMS[$(( (off + i) % 2 ))]}" "$cell" "$cfg" "$r" || fail=$((fail + 1))
    done
  done
done

harness_done "$fail"
