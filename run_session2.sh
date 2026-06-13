#!/bin/bash
# Session-2 orchestrator. Keeps the single 4090 busy back-to-back: Run 1 (priority, long)
# starts FIRST; quick GPU reprobes are slotted where the GPU would otherwise idle; HF uploads
# run in the background so they never stall training. Raw data is committed+pushed after every
# stage so partial results survive a crash. The watchdog (separate) handles the verified stop.
set -u
cd /workspace/vlm-sink-emergence
source /workspace/venv/bin/activate
export HF_HOME=/workspace/.hf_home PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
REPO=nemesismaniac/vlm-sink-emergence-ckpts
log(){ echo "[orch $(date -u +%H:%M:%S)] $*"; }

upload_bg(){ # $1=local path (uploaded to same path in repo); echoes the bg PID
  ( huggingface-cli upload "$REPO" "$1" "$1" --repo-type dataset ) >>hf_upload.log 2>&1 &
  echo $!; }

reprobe_all(){ # $1=arm $2=run_dir
  local steps; steps=$(ls "$2"/ckpt_step*.pt 2>/dev/null | sed -E 's/.*ckpt_step([0-9]+)\.pt/\1/' | sort -n | paste -sd,)
  [ -z "$steps" ] && { log "no ckpts to reprobe in $2"; return; }
  log "reprobe $1 $2 steps=$steps"
  python -u reprobe.py --arm "$1" --run_dir "$2" --ckpts "$steps" 2>&1 | sed "s/^/[reprobe $1] /"
}

commit(){ # $1=msg  (gitignore keeps *.pt and *.npz out; commits jsonl/json/summary)
  git add -A 2>/dev/null
  git commit -m "$1" 2>&1 | tail -1
  git push 2>&1 | tail -1
}

log "SESSION 2 START. free disk: $(df -h /workspace | awk 'NR==2{print $4}')"

############ RUN 1 — baseline resume to 1.0B tokens (PRIORITY, GPU busy immediately) ############
log "RUN 1: baseline resume 18287 -> 1.0B tokens"
python -u train_sinks.py --arm baseline --batch_size 128 --compile --workers 16 \
  --resume runs/baseline/ckpt_step18287.pt --resume_step 18287 --resume_tokens 174400000 \
  --max_steps 100000 --max_tokens_M 1000 --probe_every 100 --val_every 500 \
  --out_dir runs/baseline_ext --ckpt_steps 32000,48000,64000,80000,100000 \
  >runs_baseline_ext.out 2>&1
log "RUN 1 done. $(tail -1 runs_baseline_ext.out)"
U1=$(upload_bg runs/baseline_ext)            # push ckpts to HF in background
reprobe_all baseline runs/baseline_ext        # GPU stays busy while upload streams
commit "Run1: baseline extended to 1.0B (probes+reprobe)"
wait "$U1" 2>/dev/null; rm -f runs/baseline_ext/ckpt_step*.pt   # free disk only after upload lands
log "RUN 1 uploaded+cleaned. free disk: $(df -h /workspace | awk 'NR==2{print $4}')"

############ RUN 2 — seed-1, all four arms, matched 100M-token budget ############
for arm in baseline textinit g1gate sigmoid; do
  log "RUN 2: $arm seed=1 -> 100M tokens"
  python -u train_sinks.py --arm "$arm" --seed 1 --batch_size 128 --compile --workers 16 \
    --max_steps 100000 --max_tokens_M 100 --probe_every 100 --val_every 500 \
    --out_dir "runs/${arm}_seed1" --ckpt_steps 0,250,1000,2000,4000,6000 \
    >"runs_${arm}_seed1.out" 2>&1
  log "RUN 2 $arm done. $(tail -1 runs_${arm}_seed1.out)"
  UP=$(upload_bg "runs/${arm}_seed1")
  reprobe_all "$arm" "runs/${arm}_seed1"
  commit "Run2: ${arm} seed-1 to 100M (probes+reprobe)"
  wait "$UP" 2>/dev/null; rm -f "runs/${arm}_seed1"/ckpt_step*.pt
  log "RUN 2 $arm uploaded+cleaned. free disk: $(df -h /workspace | awk 'NR==2{print $4}')"
done

############ RUN 3 — reprobe the Session-1 checkpoints (still local), then free their disk ############
for d in baseline g1gate sigmoid textinit; do
  reprobe_all "$d" "runs/$d"
  US=$(upload_bg "runs/$d/reprobe")
  commit "Run3: per-head/per-position reprobe of Session-1 $d ckpts"
  wait "$US" 2>/dev/null
done
log "RUN 3 done."

############ Acceptance-gate summary (machine-readable; prose REPORT is written by Claude) ############
python -u gate_summary.py >gate_summary.txt 2>&1 || log "gate_summary failed (non-fatal)"
cat gate_summary.txt
commit "Session-2 acceptance-gate summary"

log "ALL WORK COMPLETE. free disk: $(df -h /workspace | awk 'NR==2{print $4}'). Verified stop:"
bash stop_verify.sh
