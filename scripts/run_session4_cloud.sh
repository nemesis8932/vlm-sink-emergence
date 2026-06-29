#!/bin/bash
# Session-4: two seed-2 reruns to close Gate B at n=3 (g1gate + textinit only).
# Recipe byte-identical to runs/{g1gate,textinit}_seed1 except --seed 2.
# Priority run first (g1gate), persist (upload+reprobe+commit) after EACH arm, then verified stop is
# handled SEPARATELY/interactively by the agent (exit-runbook needs generate.py quality check first).
set -u
cd /workspace/vlm-sink-emergence
source /workspace/venv/bin/activate
export WANDB_MODE=offline HF_HOME=/workspace/.hf_home PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
REPO=nemesismaniac/vlm-sink-emergence-ckpts
log(){ echo "[s4 $(date -u +%H:%M:%S)] $*"; }
upload_bg(){ ( huggingface-cli upload "$REPO" "$1" "$1" --repo-type dataset ) >>hf_upload.log 2>&1 & echo $!; }
commit(){ git add -A 2>/dev/null; git commit -m "$1" 2>&1 | tail -1; git push 2>&1 | tail -1; }
reprobe_all(){ local f; f=$(ls "$2"/ckpt_step*.pt 2>/dev/null); [ -z "$f" ] && { log "no ckpts in $2"; return; }
  local steps; steps=$(echo "$f" | sed -E 's/.*ckpt_step([0-9]+)\.pt/\1/' | sort -n | paste -sd,)
  log "reprobe $1 $2 ($steps)"; python -u reprobe.py --arm "$1" --run_dir "$2" --ckpts "$steps" 2>&1 | sed "s/^/[reprobe $1] /"; }

# priority order: g1gate first (the Sink^0.2_1 reproducibility signal), then textinit.
for arm in g1gate textinit; do
  log "R: $arm seed-2 repeated -> 100M"
  python -u train_sinks.py --arm "$arm" --data_mode repeated --seed 2 --batch_size 128 --compile --workers 16 \
    --max_steps 100000 --max_tokens_M 100 --probe_every 100 --val_every 500 \
    --out_dir "runs/${arm}_seed2" --ckpt_steps 0,250,1000,2000,4000,6000 >"runs_${arm}_seed2.out" 2>&1
  log "R $arm done: $(tail -1 runs_${arm}_seed2.out)"
  U=$(upload_bg "runs/${arm}_seed2"); reprobe_all "$arm" "runs/${arm}_seed2"
  commit "S4: ${arm} seed-2 repeated -> 100M (Gate B n=3)"; wait "$U" 2>/dev/null
  rm -f "runs/${arm}_seed2"/ckpt_step*.pt
  log "free disk after $arm cleanup: $(df -h /workspace | awk 'NR==2{print $4}')"
done
log "BOTH SEED-2 ARMS DONE. Exit-runbook handled interactively by agent (do NOT auto-stop here)."
