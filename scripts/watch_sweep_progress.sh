#!/usr/bin/env bash
# Live, one-line-per-job view of a running noise-sweep script's progress.
# Each job's tqdm bar writes to its own *.log file (carriage-return updates,
# not newlines) -- this pulls out just the current state of each and
# refreshes every 2s, instead of N tqdm bars fighting over one terminal.
#
# Usage: bash scripts/watch_sweep_progress.sh [results_dir]
# Default results_dir: outputs/noise_sweep_results/pointpillars_pipeline/runs

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
RESULTS_DIR="${1:-${ROOT}/outputs/noise_sweep_results/pointpillars_pipeline/runs}"

watch -n 2 -t "
for f in '${RESULTS_DIR}'/*.log; do
  name=\$(basename \"\$f\" .log)
  # tqdm separates updates with \r, not newline; last \r-chunk is current state.
  last=\$(tr '\r' '\n' < \"\$f\" | grep -v '^\$' | tail -1)
  if grep -q '^\[done\]' \"\$f\" 2>/dev/null; then
    status='DONE  '
  elif [ -n \"\$last\" ]; then
    status='RUN   '
  else
    status='QUEUED'
  fi
  printf '%-30s %s %s\n' \"\$name\" \"\$status\" \"\$last\"
done
"
