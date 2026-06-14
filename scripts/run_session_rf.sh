#!/bin/bash
# RF session orchestrator — fresh-arm baseline run (1B tok, 100k steps).
#
# Gate: MUST run disk_dryrun.sh first and confirm cache bounded before launching.
# Recipe LOCKED (ADR-0002): do NOT touch --batch_size/grad_accum/LR.
#
# Usage: bash scripts/run_session_rf.sh 2>&1 | tee /workspace/run_session_rf.log

set -euo pipefail
cd /workspace/vlm-sink-emergence

# Failsafe profile — watchdog_rf.sh reads these
export NAN_KILL=1
export VAL_KILL=0
export CAP_HOURS=12   # raised 8->12: streaming-fresh @ ~0.5s/step margin to finish 1B tok
export SESSION_SCRIPT=run_session_rf.sh
export COMPUTE_LOG=/workspace/vlm-sink-emergence/compute_log.jsonl
export DATA_SHUFFLE_BUFFER=500   # raw PIL ~9MB/item.
# glibc arena fragmentation = the RAM "creep" (10 workers crept 88->180G). default arenas
# 8*nproc(128)=1024 -> heavy multithread alloc/free of image buffers fragments hugely.
# cap arenas to 2 + aggressive trim -> returns freed pages to OS, kills creep -> 12 workers fits.
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=0

source /workspace/venv/bin/activate
export HF_HOME=/workspace/.hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ARM=baseline
OUT_DIR=/workspace/vlm-sink-emergence/runs/rf_fresh_${ARM}
mkdir -p "$OUT_DIR"
TRAIN_LOG="$OUT_DIR/train_log.jsonl"

echo "[rf] $(date -u) starting RF session NAN_KILL=$NAN_KILL CAP_HOURS=$CAP_HOURS"

# --- launch monitor (background) ---
python3 scripts/compute_monitor.py \
    --train_log "$TRAIN_LOG" \
    --out_dir /workspace/vlm-sink-emergence \
    --interval 45 \
    >> /workspace/vlm-sink-emergence/monitor.log 2>&1 &
MONITOR_PID=$!
echo "[rf] monitor PID=$MONITOR_PID"

# --- launch watchdog (background) ---
bash scripts/watchdog_rf.sh >> /workspace/vlm-sink-emergence/watchdog_rf.log 2>&1 &
WATCHDOG_PID=$!
echo "[rf] watchdog PID=$WATCHDOG_PID"

# --- RF training (foreground — GPU busy) ---
# --val_size 1024: fresh val drained from stream BEFORE training (see _get_data_streaming_fresh)
# --probe_every 100 for first ~5k steps; no flag to vary mid-run, leave at 100 for full run
python3 train_sinks.py \
    --arm "$ARM" \
    --vit_init pretrained \
    --data_mode fresh \
    --max_steps 100000 \
    --batch_size 128 \
    --grad_accum 1 \
    --lr_lm 4e-4 \
    --lr_mp 2e-3 \
    --lr_vit 1e-4 \
    --weight_decay 0.1 \
    --compile \
    --seed 0 \
    --val_size 1024 \
    --probe_every 100 \
    --val_every 500 \
    --workers 8 \
    --resume "${RESUME_CKPT:-}" \
    --resume_step "${RESUME_STEP:-0}" \
    --resume_tokens "${RESUME_TOKENS:-0}" \
    --out_dir "$OUT_DIR" \
    --ckpt_steps 0,250,1000,2000,4000,8000,16000,32000,64000,100000 \
    2>&1 | tee -a "$TRAIN_LOG.stdout"

echo "[rf] $(date -u) training done — persisting results"

# --- persist: commit jsonl/probes, push ---
git add "$OUT_DIR"/*.jsonl "$OUT_DIR"/run_config.json compute_log.jsonl 2>/dev/null || true
git commit -m "RF fresh arm: raw results $(date -u +%Y-%m-%dT%H:%M)" 2>/dev/null || true
git push 2>/dev/null || true

# --- upload checkpoints to HF (background) ---
python3 -c "
from huggingface_hub import HfApi
import glob, os
api = HfApi()
ckpts = glob.glob('$OUT_DIR/ckpt_step*.pt')
for ckpt in ckpts:
    api.upload_file(
        path_or_fileobj=ckpt,
        path_in_repo=f"runs/rf_fresh_baseline/{os.path.basename(ckpt)}",
        repo_id='nemesismaniac/vlm-sink-emergence-ckpts',
        repo_type='dataset',
    )
    print(f'[hf] uploaded runs/rf_fresh_baseline/{os.path.basename(ckpt)}')
" &
HF_PID=$!
echo "[rf] HF upload PID=$HF_PID"

wait $HF_PID
echo "[rf] HF upload done"

# --- verified stop ---
kill $MONITOR_PID 2>/dev/null || true
echo "[rf] $(date -u) all done — verified stop"
bash scripts/stop_verify.sh
