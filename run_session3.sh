#!/bin/bash
# Session-3 orchestrator (next remote run). Ordered run list from REMOTE-RUNBOOK.md, with
# REDUNDANT auto-stop baked in (the idle tail cost ~$1.5 last session when one stop mechanism
# silently no-op'd). Raw results are committed+pushed and checkpoints uploaded to HF after every
# run, so a crash never loses a completed arm. NEVER destroys instance 40436103 -- start/stop only.
#
#   bash run_session3.sh --dry-run     # print every train/stop/shutdown call, execute nothing
#   bash run_session3.sh               # real run
set -u

DRY_RUN=0; [ "${1:-}" = "--dry-run" ] && DRY_RUN=1
INSTANCE_ID="${INSTANCE_ID:-40436103}"     # vast.ai instance -- start/stop only, never destroy
BACKSTOP_MIN="${BACKSTOP_MIN:-30}"          # in-instance hard power-off, mechanism 2
REPO="${HF_CKPT_REPO:-nemesismaniac/vlm-sink-emergence-ckpts}"   # HF dataset for ckpts + probes
export WANDB_MODE=offline
export HF_HOME="${HF_HOME:-/workspace/.hf_home}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false

cd /workspace/vlm-sink-emergence 2>/dev/null || cd "$(dirname "$0")"
[ "$DRY_RUN" = 0 ] && source /workspace/venv/bin/activate 2>/dev/null
log(){ echo "[orch $(date -u +%H:%M:%S)] $*"; }
run(){ # echo + execute (or just echo under --dry-run)
  log "RUN: $*"
  if [ "$DRY_RUN" = 1 ]; then echo "  [dry-run] (not executed)"; else "$@"; fi
}

upload_bg(){ # $1=local path -> same path in HF dataset repo (background; never stalls training)
  if [ "$DRY_RUN" = 1 ]; then echo "  [dry-run] would: huggingface-cli upload $REPO $1 $1 --repo-type dataset"; echo 0; return; fi
  ( huggingface-cli upload "$REPO" "$1" "$1" --repo-type dataset ) >>hf_upload.log 2>&1 &
  echo $!; }

commit(){ # commit jsonl/json/summary (gitignore keeps *.pt/*.npz out); push so partial work survives
  if [ "$DRY_RUN" = 1 ]; then echo "  [dry-run] would: git add -A && git commit -m \"$1\" && git push"; return; fi
  git add -A 2>/dev/null; git commit -m "$1" 2>&1 | tail -1; git push 2>&1 | tail -1; }

reprobe_all(){ # $1=arm $2=run_dir -- per-head v/h, raw sigmoid mass, image-swap (seconds of GPU)
  local files; files=$(ls "$2"/ckpt_step*.pt 2>/dev/null)
  [ -z "$files" ] && { log "no ckpts to reprobe in $2"; return; }
  local steps; steps=$(echo "$files" | sed -E 's/.*ckpt_step([0-9]+)\.pt/\1/' | sort -n | paste -sd,)
  run python -u reprobe.py --arm "$1" --run_dir "$2" --ckpts "$steps"; }

# ---- REDUNDANT auto-stop: verified vastai stop (polls actual_status) AND in-instance shutdown ----
hard_stop(){
  log "ALL WORK COMPLETE -> redundant auto-stop (mechanism 1: vastai stop; mechanism 2: shutdown)"
  if [ "$DRY_RUN" = 1 ]; then
    echo "  [dry-run] would: bash stop_verify.sh        # vastai stop instance $INSTANCE_ID, polled until confirmed"
    echo "  [dry-run] would: shutdown -h +$BACKSTOP_MIN  # in-instance hard backstop"
    return
  fi
  bash stop_verify.sh || true
  shutdown -h +"$BACKSTOP_MIN" "sink-session3 backstop" 2>/dev/null || true
}
trap 'log "orchestrator exiting"; hard_stop' EXIT

log "SESSION 3 START dry_run=$DRY_RUN instance=$INSTANCE_ID repo=$REPO"

############ RUN 1 — finish baseline (repeated) to 1.0B; RESUME from latest ckpt, do not recompute ##
# Set these from the latest runs/baseline_ext checkpoint on HF before launching (see runbook):
R1_CKPT="${R1_CKPT:-runs/baseline_ext/ckpt_step32000.pt}"
R1_STEP="${R1_STEP:-32000}"; R1_TOK="${R1_TOK:-305000000}"
run python -u train_sinks.py --arm baseline --data_mode repeated --batch_size 128 --compile --workers 16 \
  --resume "$R1_CKPT" --resume_step "$R1_STEP" --resume_tokens "$R1_TOK" \
  --max_steps 200000 --max_tokens_M 1000 --probe_every 100 --val_every 500 \
  --out_dir runs/baseline_ext --ckpt_steps 48000,64000,80000,100000
U=$(upload_bg runs/baseline_ext); reprobe_all baseline runs/baseline_ext
commit "S3 Run1: baseline (repeated) -> 1.0B (probes+reprobe)"; [ "$DRY_RUN" = 0 ] && wait "$U" 2>/dev/null

############ RUN 2 — seed-1, four arms, repeated, 100M each (Gate B: do arms separate?) ############
for arm in baseline textinit g1gate sigmoid; do
  run python -u train_sinks.py --arm "$arm" --data_mode repeated --seed 1 --batch_size 128 --compile --workers 16 \
    --max_steps 100000 --max_tokens_M 100 --probe_every 100 --val_every 500 \
    --out_dir "runs/${arm}_seed1" --ckpt_steps 0,250,1000,2000,4000,6000
  U=$(upload_bg "runs/${arm}_seed1"); reprobe_all "$arm" "runs/${arm}_seed1"
  commit "S3 Run2: ${arm} seed-1 repeated -> 100M"; [ "$DRY_RUN" = 0 ] && wait "$U" 2>/dev/null
done

############ RUN 3 — NEW fresh-data baseline (FineVision natural pool) seed-0 -> 1.0B (Gate-A retest) #
run python -u train_sinks.py --arm baseline --data_mode fresh --seed 0 --batch_size 128 --compile --workers 16 \
  --max_steps 200000 --max_tokens_M 1000 --probe_every 100 --val_every 500 \
  --out_dir runs/baseline_fresh --ckpt_steps 0,250,1000,2000,4000,8000,12000,16000,24000,48000,80000,100000
U=$(upload_bg runs/baseline_fresh); reprobe_all baseline runs/baseline_fresh
commit "S3 Run3: fresh-data baseline (FineVision) -> 1.0B (Gate-A retest)"; [ "$DRY_RUN" = 0 ] && wait "$U" 2>/dev/null

############ RUN 4 — reprobe any remaining Session-1/2 ckpts still local ############
for d in baseline g1gate sigmoid textinit; do
  [ -d "runs/$d" ] && { reprobe_all "$d" "runs/$d"; U=$(upload_bg "runs/$d/reprobe"); commit "S3 Run4: reprobe $d"; [ "$DRY_RUN" = 0 ] && wait "$U" 2>/dev/null; }
done

run python -u gate_summary.py
commit "S3 acceptance-gate summary"
log "DONE. (EXIT trap fires hard_stop)"
