#!/bin/bash
# Adaptive poller for the textinit seed-2 RETRY. 5-min -> 15-min once healthy
# (step>=2000, loss finite, no errors). Failures + completion emit immediately.
cd /workspace/vlm-sink-emergence
INTERVAL=300; GRAD=${HOLD5:-0}   # HOLD5=1 -> pin 5-min, no graduation
errgrep='Traceback|OOM|out of memory|CUDA error|Gateway Time-out|NaN|nan,|Killed|RuntimeError|AssertionError|MemoryError|HfHubHTTPError'
A=runs_textinit_seed2.out
while true; do
  loss=$(grep -E '\] step [0-9]+ loss' "$A" 2>/dev/null | tail -1)
  probe=$(grep -E '\[probe' "$A" 2>/dev/null | tail -1)
  step=$(sed -E 's/.*\] step ([0-9]+) loss.*/\1/' <<<"$loss")
  lv=$(sed -E 's/.* loss ([0-9.]+) .*/\1/' <<<"$loss")
  if err=$(grep -hE "$errgrep" runs_textinit_seed2.out runs_textinit_retry.out 2>/dev/null | tail -1); [ -n "$err" ]; then
    echo "FAIL @ $(date -u +%H:%M): $err"
  fi
  if grep -q "TEXTINIT SEED-2 DONE (retry)" runs_textinit_retry.out 2>/dev/null; then
    echo "COMPLETE @ $(date -u +%H:%M): textinit seed-2 done -> exit-runbook"; exit 0
  fi
  if [ "$GRAD" -eq 0 ] && [ -n "$step" ] && [ "$step" -ge 2000 ] && [ -n "$lv" ] && [ -z "$err" ]; then
    GRAD=1; INTERVAL=900
    echo "GRADUATED @ $(date -u +%H:%M): healthy (step $step loss $lv) -> 15-min. ${probe#*] }"
  else
    echo "TICK @ $(date -u +%H:%M) [$((INTERVAL/60))m]: ${loss#*] } | ${probe#*] }"
  fi
  sleep "$INTERVAL"
done
