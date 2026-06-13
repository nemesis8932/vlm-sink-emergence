#!/bin/bash
# Cost backstop for the Session-3 wrapper. The wrapper has its own EXIT-trap stop; this catches
# the case where it is SIGKILLed / the box hangs. Stops 15 min after the wrapper disappears, or
# at a 16h absolute cap. Never stops while run_session3_cloud.sh is alive (long train/upload safe).
cd /workspace/vlm-sink-emergence
DEADLINE=$(( $(date +%s) + 16*3600 )); gone=0
while true; do
  sleep 60
  if pgrep -f run_session3_cloud.sh >/dev/null 2>&1; then gone=0; else gone=$((gone+1)); fi
  if [ "$gone" -ge 15 ]; then echo "[wd2] wrapper absent 15min @ $(date -u) -> stop"; bash stop_verify.sh; exit 0; fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then echo "[wd2] 16h cap @ $(date -u) -> stop"; bash stop_verify.sh; exit 0; fi
done
