#!/bin/bash
# RF-session watchdog. Reads failsafe profile from env; stops instance on trigger.
#
# Failsafe profile (set by run_session before launch):
#   NAN_KILL=1      stop on NaN/Inf loss in compute_log.jsonl (default 1)
#   VAL_KILL=0      stop on val-loss divergence — DISABLED for RF (Director judges)
#   CAP_HOURS=8     hard wall-clock cap (default 8)
#
# Stops on: NaN/Inf loss (if NAN_KILL=1), 8h cap, orchestrator absent 10 consecutive ticks.
# Logs but does NOT stop: util~0 hang, disk pressure, loss divergence (Director judges).
#
# Usage: bash scripts/watchdog_rf.sh >> /workspace/watchdog_rf.log 2>&1 &

cd /workspace/vlm-sink-emergence || { echo "[watchdog_rf] cannot cd to workspace"; exit 1; }

NAN_KILL=${NAN_KILL:-1}
VAL_KILL=${VAL_KILL:-0}
CAP_HOURS=${CAP_HOURS:-8}
COMPUTE_LOG=${COMPUTE_LOG:-/workspace/vlm-sink-emergence/compute_log.jsonl}
SESSION_SCRIPT=${SESSION_SCRIPT:-run_session_rf.sh}

START_EPOCH=$(date +%s)
HARD_DEADLINE=$(( START_EPOCH + CAP_HOURS * 3600 ))
gone=0

echo "[watchdog_rf] start $(date -u) NAN_KILL=$NAN_KILL VAL_KILL=$VAL_KILL CAP_HOURS=$CAP_HOURS"

while true; do
    sleep 60

    NOW=$(date +%s)

    # 1. Hard cap
    if [ "$NOW" -ge "$HARD_DEADLINE" ]; then
        echo "[watchdog_rf] ${CAP_HOURS}h cap hit @ $(date -u) -> verified stop"
        bash scripts/stop_verify.sh; exit 0
    fi

    # 2. NaN/Inf kill — reads loss field from last 5 compute_log.jsonl entries
    if [ "$NAN_KILL" = "1" ] && [ -f "$COMPUTE_LOG" ]; then
        LAST_LOSS=$(tail -n 5 "$COMPUTE_LOG" | python3 -c "
import sys, json, math
for line in sys.stdin:
    try:
        r = json.loads(line)
        v = r.get('loss')
        if v is not None:
            print(v)
    except: pass
" 2>/dev/null | tail -1)
        if [ -n "$LAST_LOSS" ]; then
            IS_BAD=$(python3 -c "
import sys, math
try:
    v = float('$LAST_LOSS')
    print('1' if (math.isnan(v) or math.isinf(v)) else '0')
except:
    print('0')
" 2>/dev/null)
            if [ "$IS_BAD" = "1" ]; then
                echo "[watchdog_rf] NaN/Inf loss=$LAST_LOSS @ $(date -u) -> verified stop"
                bash scripts/stop_verify.sh; exit 1
            fi
        fi
    fi

    # 3. Orchestrator-gone check (10 consecutive absent ticks)
    if pgrep -f "$SESSION_SCRIPT" >/dev/null 2>&1; then
        gone=0
    else
        gone=$((gone + 1))
        echo "[watchdog_rf] orchestrator absent tick $gone @ $(date -u)"
    fi
    if [ "$gone" -ge 10 ]; then
        echo "[watchdog_rf] orchestrator absent 10min @ $(date -u) -> verified stop"
        bash scripts/stop_verify.sh; exit 0
    fi

    # 4. Monitor-only: log anomalies but do NOT kill (Director judges)
    if [ -f "$COMPUTE_LOG" ]; then
        GPU_UTIL=$(tail -n 1 "$COMPUTE_LOG" | python3 -c "
import sys, json
try:
    r = json.loads(sys.stdin.read())
    print(r.get('gpu_util_pct', ''))
except: pass
" 2>/dev/null)
        DISK_AVAIL=$(tail -n 1 "$COMPUTE_LOG" | python3 -c "
import sys, json
try:
    r = json.loads(sys.stdin.read())
    print(r.get('disk_avail_mb', ''))
except: pass
" 2>/dev/null)
        if [ -n "$GPU_UTIL" ] && python3 -c "exit(0 if float('$GPU_UTIL') < 5 else 1)" 2>/dev/null; then
            echo "[watchdog_rf] WARN: gpu_util=${GPU_UTIL}% (hang?) @ $(date -u) -- logging, not killing"
        fi
        if [ -n "$DISK_AVAIL" ] && python3 -c "exit(0 if float('$DISK_AVAIL') < 5000 else 1)" 2>/dev/null; then
            echo "[watchdog_rf] WARN: disk_avail=${DISK_AVAIL}MB @ $(date -u) -- logging, not killing"
        fi
    fi
done
