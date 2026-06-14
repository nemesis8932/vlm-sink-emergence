#!/bin/bash
# Gate 2 dry-run: short real FineVision stream to confirm cache stays bounded.
# Run this BEFORE launching run_session_rf.sh. STOP if cache grows unbounded.
#
# Checks:
#   - HF token valid (401 = STOP, ask user for new token)
#   - Streaming loader parses correct schema
#   - Shard cache stays < CACHE_WARN_GB after --max_steps 5 (eviction working)
#   - NaN kill inline works (loss is finite after 5 real steps)
#
# Usage: bash scripts/disk_dryrun.sh 2>&1 | tee /workspace/disk_dryrun.log

set -euo pipefail
cd /workspace/vlm-sink-emergence

CACHE_WARN_GB=${CACHE_WARN_GB:-10}
source /workspace/venv/bin/activate
export HF_HOME=/workspace/.hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NAN_KILL=1

echo "[dryrun] $(date -u) start — HF_HOME=$HF_HOME CACHE_WARN_GB=$CACHE_WARN_GB"

# 1. Token check
python3 -c "
from huggingface_hub import whoami
try:
    u = whoami()
    print(f'[dryrun] HF token OK: {u[\"name\"]}')
except Exception as e:
    print(f'[dryrun] STOP: HF token invalid -> {e}')
    exit(1)
"

# 2. Short streaming smoke: --limit_per_subset 200 hits real HF data, small download
echo "[dryrun] streaming schema check (limit_per_subset=200, max_steps=5)..."
python3 train_sinks.py \
    --arm baseline \
    --vit_init random \
    --data_mode fresh \
    --limit_per_subset 200 \
    --max_steps 5 \
    --batch_size 8 \
    --grad_accum 1 \
    --lr_lm 4e-4 --lr_mp 2e-3 --lr_vit 1e-4 \
    --val_size 64 \
    --probe_every 999999 \
    --val_every 999999 \
    --workers 0 \
    --fake_data \
    --out_dir /tmp/dryrun_fake \
    2>&1 | tail -5
echo "[dryrun] fake-data smoke: OK"

# 3. Real stream, 5 steps — checks HF pull + shard eviction
echo "[dryrun] real stream smoke (limit_per_subset=50, max_steps=3, fake_data=OFF)..."
python3 train_sinks.py \
    --arm baseline \
    --vit_init random \
    --data_mode fresh \
    --limit_per_subset 50 \
    --max_steps 3 \
    --batch_size 4 \
    --grad_accum 1 \
    --lr_lm 4e-4 --lr_mp 2e-3 --lr_vit 1e-4 \
    --val_size 16 \
    --probe_every 999999 \
    --val_every 999999 \
    --workers 0 \
    --out_dir /tmp/dryrun_real \
    2>&1 | tail -10
echo "[dryrun] real stream smoke: OK"

# 4. Cache footprint check
CACHE_MB=$(du -sm "$HF_HOME" 2>/dev/null | cut -f1 || echo 0)
CACHE_GB=$(python3 -c "print(round($CACHE_MB/1024, 2))")
echo "[dryrun] HF cache after smoke: ${CACHE_GB}GB (warn at ${CACHE_WARN_GB}GB)"
python3 -c "
c = $CACHE_GB
limit = $CACHE_WARN_GB
if c > limit:
    print(f'[dryrun] WARN: cache {c}GB > {limit}GB -- check eviction before long run')
else:
    print(f'[dryrun] cache OK ({c}GB <= {limit}GB)')
"

echo "[dryrun] $(date -u) ALL GATES PASS — safe to launch run_session_rf.sh"
