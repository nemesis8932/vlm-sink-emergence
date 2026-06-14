#!/bin/bash
# Cost backstop. Stops the instance once the orchestrator process is gone (clean finish OR
# crash) for 10 consecutive minutes, or after a 14h absolute cap -- whichever first. Never
# stops while run_session2.sh is alive, so long blocking train/upload steps are safe.
cd /workspace/vlm-sink-emergence
HARD_DEADLINE=$(( $(date +%s) + 14*3600 ))
gone=0
while true; do
  sleep 60
  if pgrep -f run_session2.sh >/dev/null 2>&1; then gone=0; else gone=$((gone+1)); fi
  if [ "$gone" -ge 10 ]; then
    echo "[watchdog] orchestrator absent 10min @ $(date -u) -> verified stop"
    bash stop_verify.sh; exit 0
  fi
  if [ "$(date +%s)" -ge "$HARD_DEADLINE" ]; then
    echo "[watchdog] 14h cap @ $(date -u) -> verified stop"
    bash stop_verify.sh; exit 0
  fi
done
