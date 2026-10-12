#!/usr/bin/env bash
# SERVER (AutoDL): package the pilot metrics + logs to send back for adjudication.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
tar czf results.tar.gz results/pilot_metrics.json runs/R*_*.log 2>/dev/null \
  $(ls -d runs/R*_*/*/vis_data 2>/dev/null) || tar czf results.tar.gz results/pilot_metrics.json
echo "DONE -> $ROOT/results.tar.gz  (download this and send it back)"
du -h results.tar.gz
echo "--- quick peek (paste this if upload is slow) ---"
cat results/pilot_metrics.json 2>/dev/null || echo "no metrics json (check runs/*.log for errors)"
