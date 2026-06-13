#!/bin/bash
# Idempotent VERIFIED instance stop. In-instance `vastai stop` is unreliable (it can print
# success while the instance keeps running -- see memory note from Session 1), so we issue it
# and then poll actual_status, retrying until confirmed stopped.
# Guard: while /workspace/vlm-sink-emergence/NO_AUTOSTOP exists, stopping is deferred (used to
# hold the box up until the export bundle is built + shipped; the export step removes the flag).
if [ -f /workspace/vlm-sink-emergence/NO_AUTOSTOP ]; then
  echo "[stop] NO_AUTOSTOP present -> deferring stop"; exit 0
fi
source /workspace/venv/bin/activate 2>/dev/null
status() {
  vastai show instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" --raw 2>/dev/null \
    | python3 -c "import sys,json; print(json.load(sys.stdin).get('actual_status',''))" 2>/dev/null
}
for i in $(seq 1 8); do
  st=$(status)
  echo "[stop] attempt $i status=$st $(date -u +%H:%M:%S)"
  if [ "$st" = "stopped" ] || [ "$st" = "exited" ]; then echo "[stop] CONFIRMED stopped"; exit 0; fi
  vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" 2>&1 | sed 's/^/[stop] /'
  sleep 60
done
echo "[stop] WARNING: could not confirm stop after 8 attempts -- stop manually from dashboard"
exit 1
