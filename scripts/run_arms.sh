#!/bin/bash
# Chains the remaining arms after the baseline finishes. GPU stays busy unattended.
set -u
cd /workspace/vlm-sink-emergence
source /workspace/venv/bin/activate
export HF_HOME=/workspace/.hf_home PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASELINE_PID=$1

while kill -0 "$BASELINE_PID" 2>/dev/null; do sleep 20; done
echo "baseline done at $(date -u)"

python -u train_sinks.py --arm textinit --batch_size 128 --max_steps 100000 --max_hours 0.55 \
  --probe_every 100 --val_every 500 --workers 16 --compile --out_dir runs/textinit \
  > runs_textinit.out 2>&1
echo "textinit done at $(date -u)"

python -u train_sinks.py --arm g1gate --batch_size 128 --max_steps 100000 --max_hours 1.0 \
  --probe_every 100 --val_every 500 --workers 16 --compile --out_dir runs/g1gate \
  > runs_g1gate.out 2>&1
echo "g1gate done at $(date -u)"

python -u train_sinks.py --arm sigmoid --batch_size 128 --max_steps 100000 --max_hours 1.0 \
  --probe_every 100 --val_every 500 --workers 16 --compile --out_dir runs/sigmoid \
  > runs_sigmoid.out 2>&1
echo "sigmoid done at $(date -u)"
echo "ALL ARMS COMPLETE"
