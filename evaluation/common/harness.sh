# Shared helpers for the collect.sh scripts. Source it, then call harness_init.
#
# Common options (parsed by harness_init, removed from "$@" via HARGS):
#   --dry-run   print every run instead of executing it
#   --smoke     1 run per cell, 2 queries per run, local MLflow store
#   --runs N    repetitions per cell (run 1 is a warm-up and is dropped in analysis)
#
# Environment:
#   RESULTS_DIR           output root (default: <repo>/results)
#   MLFLOW_TRACKING_URI   tracking server; required unless --smoke or a local store is used

set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DRY_RUN=0; SMOKE=0; RUNS=""; HARGS=()

harness_init() {
  # usage: harness_init <experiment> <default_runs> "$@"
  EXPERIMENT=$1; local default_runs=$2; shift 2
  while [ $# -gt 0 ]; do
    case $1 in
      --dry-run) DRY_RUN=1 ;;
      --smoke) SMOKE=1 ;;
      --runs) RUNS=$2; shift ;;
      *) HARGS+=("$1") ;;
    esac
    shift
  done
  RUNS=${RUNS:-$default_runs}
  MAX_QUERIES_ARGS=()
  if [ "$SMOKE" = 1 ]; then
    RUNS=1
    MAX_QUERIES_ARGS=(--max-queries 2)
  fi
  cd "$ROOT"
}

harness_outdir() {
  # usage: harness_outdir <device>; sets OUT, LOG, SUM and exports SUITE_OUTPUT_DIR
  local device=$1 stamp
  OUT="${RESULTS_DIR:-$ROOT/results}/$EXPERIMENT/$device"
  [ "$SMOKE" = 1 ] && OUT="$OUT/smoke"
  export SUITE_OUTPUT_DIR="$OUT"
  stamp=$(date '+%Y%m%d-%H%M%S')
  LOG="$OUT/logs/collect_${stamp}.log"
  SUM="$OUT/logs/collect_${stamp}.tsv"
  if [ "$DRY_RUN" = 0 ]; then
    mkdir -p "$OUT/logs"
    printf 'label\trc\tseconds\tcsv_rows\tspans\n' > "$SUM"
    harness_provenance >> "$LOG"
    pip freeze > "$OUT/logs/packages_${stamp}.txt" 2>/dev/null
  fi
}

harness_keep_awake() {
  # Re-exec under caffeinate on macOS so the machine cannot sleep mid-collection.
  if [ "$(uname)" = "Darwin" ] && [ -z "${_CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
    export _CAFFEINATED=1
    local script; script="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; shift
    exec caffeinate -dimsu bash "$script" "$@"
  fi
}

harness_tracking() {
  # usage: harness_tracking [local]; a local sqlite store under $OUT when asked
  # for (or in smoke mode), otherwise MLFLOW_TRACKING_URI must be set.
  if [ "${1:-}" = local ] || [ "$SMOKE" = 1 ]; then
    export MLFLOW_TRACKING_URI="sqlite:///$OUT/mlflow.db"
    unset MLFLOW_TRACKING_USERNAME MLFLOW_TRACKING_PASSWORD
    EXP_ID=0
  elif [ -z "${MLFLOW_TRACKING_URI:-}" ] && [ "$DRY_RUN" = 0 ]; then
    echo "MLFLOW_TRACKING_URI is not set (use --smoke for a local store)" >&2
    exit 2
  fi
  EXP_ID=${EXP_ID:-${MLFLOW_EXPERIMENT:-0}}
}

harness_provenance() {
  echo "# experiment : $EXPERIMENT"
  echo "# started    : $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "# git        : $(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)$(test -n "$(git -C "$ROOT" status --porcelain 2>/dev/null)" && echo ' (dirty)')"
  echo "# host       : $(hostname)"
  echo "# runs       : $RUNS$( [ "$SMOKE" = 1 ] && echo ' (smoke)')"
}

log() {
  local m="[$(date '+%m-%d %H:%M:%S')] $*"
  echo "$m"
  [ "$DRY_RUN" = 0 ] && [ -n "${LOG:-}" ] && echo "$m" >> "$LOG"
  return 0
}

run_cell() {
  # usage: run_cell <label> <command...>
  # Runs the command unless the cell completed before; records rc/rows/spans.
  local lab=$1; shift
  if [ "$DRY_RUN" = 1 ]; then
    echo "$*"
    return 0
  fi
  if [ -f "$OUT/$lab.done" ]; then
    log "  [skip] $lab (done)"
    return 0
  fi
  rm -f "$OUT/$lab.csv" "$OUT/${lab}_outputs.jsonl"
  local outfile start rc secs rows spans
  outfile=$(mktemp)
  start=$(date +%s)
  "$@" 2>&1 | tee "$outfile"
  rc=${PIPESTATUS[0]}
  secs=$(( $(date +%s) - start ))
  spans=$(sed -n 's/^\[[a-z]*\] spans emitted: //p' "$outfile" | tail -1)
  rm -f "$outfile"
  rows=0
  [ -f "$OUT/$lab.csv" ] && rows=$(wc -l < "$OUT/$lab.csv" | tr -d ' ')
  printf '%s\t%s\t%s\t%s\t%s\n' "$lab" "$rc" "$secs" "$rows" "${spans:-}" >> "$SUM"
  log "  $lab rc=$rc ${secs}s rows=$rows spans=${spans:-n/a}"
  [ "$rc" -eq 0 ] && [ "$rows" -gt 0 ] && touch "$OUT/$lab.done"
}

harness_done() {
  # usage: harness_done <failures>
  [ "$DRY_RUN" = 1 ] && return 0
  log "done: $1 failed run(s); results in $OUT"
  [ "$1" -eq 0 ]
}
