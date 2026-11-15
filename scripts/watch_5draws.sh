#!/usr/bin/env bash
# Live view of the revision-5draws run, refreshed every 5 s (Ctrl-C stops the view, not the run):
#   - the latest tqdm bar of every host network (one line each; "done" when finished);
#   - the main log: stage messages and the latest tqdm bar of the delay and pose-error stages.
# Usage (from anywhere): bash /mnt/external/workspace/dhia/AlignFormer/scripts/watch_5draws.sh
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
MAIN=outputs/paper/logs/revision_5draws.log
while true; do
  clear
  echo "revision-5draws   $(date +%H:%M:%S)   (Ctrl-C closes this view; the run continues)"
  echo
  echo "host networks (sigma 1 and 2, draws 1-4):"
  for f in outputs/paper/hosts_5draws/*.log outputs/paper/hosts_v2xreal_5draws/*.log; do
    [[ -f "$f" ]] || continue
    line="$(tr '\r' '\n' < "$f" | grep -E 'cond/s|done|FAILED' | tail -1 | sed 's/^.*conditions)://')"
    printf "  %-22s %s\n" "$(basename "$f" .log)" "${line:-starting}"
  done
  echo
  echo "main log:"
  if [[ -f "${MAIN}" ]]; then
    grep -E '^\[|skipped|FAILED|finished|stopping' "${MAIN}" | tail -n 8 | sed 's/^/  /'
    bar="$(tr '\r' '\n' < "${MAIN}" | grep -E '[0-9]+%\|' | tail -1)"
    [[ -n "${bar}" ]] && echo "  ${bar}"
  else
    echo "  (not started)"
  fi
  sleep 5
done
