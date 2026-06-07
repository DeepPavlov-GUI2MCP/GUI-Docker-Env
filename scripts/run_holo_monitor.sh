#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${HOLO_MONITOR_SESSION:-holo-eval-monitor}"

EVAL_JSON="${1:-evaluation_examples/test_synthetic_writer_gpt54_all.json}"
RESULTS_TARGET="${2:-results/results_holo_gpt54_writer_traces_*}"
TITLE="${3:-Holo Eval Monitor}"

MONITOR_ARGS=(
  --eval-json "${EVAL_JSON}"
  --title "${TITLE}"
  --refresh-seconds "${REFRESH_SECONDS:-2}"
)

if [[ "${RESULTS_TARGET}" == *"*"* ]]; then
  MONITOR_ARGS+=(--results-glob "${RESULTS_TARGET}")
else
  MONITOR_ARGS+=(--results-dir "${RESULTS_TARGET}")
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux kill-session -t "${SESSION}"
fi

tmux new-session -d -s "${SESSION}" -n monitor \
  "source ${ROOT}/../.venv/bin/activate 2>/dev/null || true; cd ${ROOT} && exec python3 scripts/monitor_holo_eval.py ${MONITOR_ARGS[*]@Q}"

echo "Started monitor in tmux session: ${SESSION}"
echo "Attach: tmux attach -t ${SESSION}"
