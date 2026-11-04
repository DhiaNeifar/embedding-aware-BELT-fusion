#!/usr/bin/env bash
# Shows exactly which (pipeline, method, noise dimension) combos have results
# in summary.csv, and which noise values are covered for each.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

python3 -c "
import csv
rows = list(csv.DictReader(open('${ROOT}/outputs/noise_sweep_results/summary.csv')))
d = {}
for r in rows:
    d.setdefault((r['pipeline'], r['method'], r['sweep_dimension']), set()).add(r['sweep_value'])
for k in sorted(d.keys()):
    print(k, '->', sorted(d[k], key=float))
"
