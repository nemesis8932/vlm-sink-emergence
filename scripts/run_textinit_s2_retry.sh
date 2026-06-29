#!/bin/bash
# Retry textinit seed-2 (first attempt died on a transient HF 504 fetching the_cauldron tree).
# Same recipe as run_session4_cloud.sh, single arm, with persist after.
set -u
cd /workspace/vlm-sink-emergence
source /workspace/venv/bin/activate
export WANDB_MODE=offline HF_HOME=/workspace/.hf_home PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
REPO=nemesismaniac/vlm-sink-emergence-ckpts
log(){ echo "[s4r $(date -u +%H:%M:%S)] $*"; }
commit(){ git add -A 2>/dev/null; git commit -m "$1" 2>&1 | tail -1; git push 2>&1 | tail -1; }
reprobe_all(){ local f; f=$(ls "$2"/ckpt_step*.pt 2>/dev/null); [ -z "$f" ] && { log "no ckpts in $2"; return; }
  local steps; steps=$(echo "$f" | sed -E 's/.*ckpt_step([0-9]+)\.pt/\1/' | sort -n | paste -sd,)
  log "reprobe $1 $2 ($steps)"; python -u reprobe.py --arm "$1" --run_dir "$2" --ckpts "$steps" 2>&1 | sed "s/^/[reprobe $1] /"; }

rm -rf runs/textinit_seed2   # clear the empty failed dir
log "R: textinit seed-2 repeated -> 100M (retry)"
python -u train_sinks.py --arm textinit --data_mode repeated --seed 2 --batch_size 128 --compile --workers 16 \
  --max_steps 100000 --max_tokens_M 100 --probe_every 100 --val_every 500 \
  --out_dir runs/textinit_seed2 --ckpt_steps 0,250,1000,2000,4000,6000 >runs_textinit_seed2.out 2>&1
rc=$?
log "R textinit exit=$rc: $(tail -1 runs_textinit_seed2.out)"
if [ "$rc" -ne 0 ]; then log "TEXTINIT FAILED AGAIN (rc=$rc) -- not persisting; investigate"; exit "$rc"; fi
( huggingface-cli upload "$REPO" runs/textinit_seed2 runs/textinit_seed2 --repo-type dataset ) >>hf_upload.log 2>&1 &
U=$!
reprobe_all textinit runs/textinit_seed2
commit "S4: textinit seed-2 repeated -> 100M (Gate B n=3, retry after 504)"
wait "$U" 2>/dev/null
rm -f runs/textinit_seed2/ckpt_step*.pt
log "TEXTINIT SEED-2 DONE (retry). free disk: $(df -h /workspace | awk 'NR==2{print $4}')"
