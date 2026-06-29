#!/bin/bash
# Adaptive status poller for session-4 seed-2 reruns.
# Starts at 5-min ticks; graduates to 15-min once the run is clearly healthy
# (latest step >= 2000, loss finite, no error markers). Each tick emits ONE compact
# status line. Failures and the completion marker emit immediately.
cd /workspace/vlm-sink-emergence
INTERVAL=300; GRAD=0
errgrep='Traceback|OOM|out of memory|CUDA error|NaN|nan,|Killed|RuntimeError|AssertionError|MemoryError'
while true; do
  active=$(ls -t runs_g1gate_seed2.out runs_textinit_seed2.out 2>/dev/null | head -1)
  loss=$(grep -E '\] step [0-9]+ loss' "$active" 2>/dev/null | tail -1)
  probe=$(grep -E '\[probe' "$active" 2>/dev/null | tail -1)
  step=$(sed -E 's/.*\] step ([0-9]+) loss.*/\1/' <<<"$loss")
  lv=$(sed -E 's/.* loss ([0-9.]+) .*/\1/' <<<"$loss")
  # immediate failure surfacing
  if err=$(grep -hE "$errgrep" runs_*_seed2.out runs_session4.out 2>/dev/null | tail -1); [ -n "$err" ]; then
    echo "FAIL @ $(date -u +%H:%M): $err"
  fi
  # completion
  if grep -q "BOTH SEED-2 ARMS DONE" runs_session4.out 2>/dev/null; then
    echo "COMPLETE @ $(date -u +%H:%M): both seed-2 arms done -> exit-runbook"; exit 0
  fi
  # graduation check (only while still at 5-min)
  if [ "$GRAD" -eq 0 ] && [ -n "$step" ] && [ "$step" -ge 2000 ] && [ -n "$lv" ] && [ -z "$err" ]; then
    GRAD=1; INTERVAL=900
    echo "GRADUATED @ $(date -u +%H:%M): healthy (step $step loss $lv) -> 15-min ticks. ${active##*/} | $probe"
  else
    echo "TICK @ $(date -u +%H:%M) [$((INTERVAL/60))m]: ${active##*/} | ${loss#*] } | ${probe#*] }"
  fi
  sleep "$INTERVAL"
done
