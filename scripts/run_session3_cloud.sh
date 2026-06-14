#!/bin/bash
# Session-3 go/no-go layer (the runbook's role) wrapped around the live Run-1 + run_session3.sh
# commands. Differences from the bare run_session3.sh, per plans/session3-cloud.md:
#   * Run-1 (baseline_ext, repeated) is ALREADY LIVE -> we WAIT for it, never relaunch it (G0.1).
#   * G0.2 HARD GATE: validate the FRESH FineVision loader on REAL streamed data before any fresh spend.
#   * RF (fresh-1B) is HARD-SKIPPED on this instance: the full FineVision pool is ~1.49 TB vs an
#     80 GB disk (measured). Fresh-1B needs a streaming loader or a multi-TB/volume instance.
#     We still run the cheap G0.2 parse-check and record the blocker for the Mac to fix.
#   * R2 (seed-1 x 4 arms, repeated, 100M) + R3 (reprobe) run normally -> Gate B + un-pooled reprobe.
set -u
cd /workspace/vlm-sink-emergence
source /workspace/venv/bin/activate
export WANDB_MODE=offline HF_HOME=/workspace/.hf_home PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
REPO=nemesismaniac/vlm-sink-emergence-ckpts
log(){ echo "[s3 $(date -u +%H:%M:%S)] $*"; }
upload_bg(){ ( huggingface-cli upload "$REPO" "$1" "$1" --repo-type dataset ) >>hf_upload.log 2>&1 & echo $!; }
commit(){ git add -A 2>/dev/null; git commit -m "$1" 2>&1 | tail -1; git push 2>&1 | tail -1; }
reprobe_all(){ local f; f=$(ls "$2"/ckpt_step*.pt 2>/dev/null); [ -z "$f" ] && { log "no ckpts in $2"; return; }
  local steps; steps=$(echo "$f" | sed -E 's/.*ckpt_step([0-9]+)\.pt/\1/' | sort -n | paste -sd,)
  log "reprobe $1 $2 ($steps)"; python -u reprobe.py --arm "$1" --run_dir "$2" --ckpts "$steps" 2>&1 | sed "s/^/[reprobe $1] /"; }
hard_stop(){ log "EXIT -> redundant verified stop"; bash stop_verify.sh || true; shutdown -h +20 "s3 backstop" 2>/dev/null || true; }
trap hard_stop EXIT

############ G0.1 — wait for the LIVE Run-1 to finish in place, then capture the narrow Gate-A ######
log "waiting for live Run-1 (baseline_ext) to finish to 1.0B ..."
while pgrep -f 'train_sinks.py.*baseline_ext' >/dev/null 2>&1; do sleep 60; done
log "Run-1 finished. $(tail -1 runs_baseline_ext.out)"
python3 - <<'PY' > gateA_narrow.txt 2>&1
import json
seen={}
for fn in ('runs/baseline/probes.jsonl','runs/baseline_ext/probes.jsonl'):
    try:
        for l in open(fn): r=json.loads(l); seen[r['tokens_seen']]=r
    except FileNotFoundError: pass
ps=[seen[t] for t in sorted(seen)]; p=ps[-1]; s=p['summary']
fc=next((q['tokens_seen']/1e6 for q in ps if q['summary']['max_attn_pos0']>0.2),None)
print("GATE A (narrow): baseline seed-0, repeated data, full curve to 1.0B")
print(f"  final {p['tokens_seen']/1e6:.1f}M tok  step {p['step']}")
print(f"  Sink^0.2={s['sink_eps0.2']:.4f} Sink^0.3={s['sink_eps0.3']:.4f} Sink^0.4={s['sink_eps0.4']:.4f}")
print(f"  max mean-attn->pos0={s['max_attn_pos0']:.4f}  mean={s['mean_attn_pos0']:.4f}")
print(f"  first head crosses 0.2: {str(fc)+'M' if fc else 'NEVER through 1.0B'}")
print(f"  v_ratio={s['v_ratio_pos0']:.3f} h_ratio={s['h_ratio_pos0']:.3f}")
PY
cat gateA_narrow.txt
U=$(upload_bg runs/baseline_ext); reprobe_all baseline runs/baseline_ext
commit "S3 G0.1: baseline (repeated) reached 1.0B; narrow Gate-A captured"
wait "$U" 2>/dev/null; rm -f runs/baseline_ext/ckpt_step*.pt
log "free disk after Run-1 cleanup: $(df -h /workspace | awk 'NR==2{print $4}')"

############ G0.2 — REAL-data fresh-loader validation (cheap streamed smoke; HARD GATE for any fresh) #
log "G0.2: fresh FineVision loader smoke (streamed, real schema)"
FRESH_OK=0
if timeout 1800 python -u train_sinks.py --arm baseline --data_mode fresh --limit_per_subset 48 \
     --batch_size 16 --workers 8 --max_steps 120 --probe_every 60 --val_every 60 \
     --val_size 64 --probe_n 16 --out_dir /tmp/fresh_smoke >runs_fresh_smoke.out 2>&1 \
   && grep -q "image-swap self-check OK" runs_fresh_smoke.out; then
  FRESH_OK=1; log "G0.2 PASS: $(grep 'image-swap self-check OK' runs_fresh_smoke.out | tail -1)"
else
  log "G0.2 FAIL (see runs_fresh_smoke.out tail):"; tail -15 runs_fresh_smoke.out
fi
cp runs_fresh_smoke.out runs/fresh_smoke.out 2>/dev/null; commit "S3 G0.2: fresh-loader real-data smoke (FRESH_OK=$FRESH_OK)"

############ R2 — seed-1, four arms, repeated, 100M each (Gate B). No disk/loader risk. ############
for arm in baseline textinit g1gate sigmoid; do
  log "R2: $arm seed-1 repeated -> 100M"
  python -u train_sinks.py --arm "$arm" --data_mode repeated --seed 1 --batch_size 128 --compile --workers 16 \
    --max_steps 100000 --max_tokens_M 100 --probe_every 100 --val_every 500 \
    --out_dir "runs/${arm}_seed1" --ckpt_steps 0,250,1000,2000,4000,6000 >"runs_${arm}_seed1.out" 2>&1
  log "R2 $arm done: $(tail -1 runs_${arm}_seed1.out)"
  U=$(upload_bg "runs/${arm}_seed1"); reprobe_all "$arm" "runs/${arm}_seed1"
  commit "S3 R2: ${arm} seed-1 repeated -> 100M"; wait "$U" 2>/dev/null; rm -f "runs/${arm}_seed1"/ckpt_step*.pt
done

############ RF — fresh-1B: HARD-SKIPPED on this instance (1.49 TB pool vs 80 GB disk) ############
cat > RF-BLOCKED.md <<'MD'
# RF (fresh-data baseline 1B) — BLOCKED on instance 40436103, not run this session

**Why:** the FRESH FineVision pool (11 natural-image configs in `data/mixes.py:FRESH`) totals
**~1.49 TB** to download (measured via `load_dataset_builder(...).info.download_size`), and the
loader (`mixes.load_mix`, `limit_per_subset=None`) does a **non-streaming full `load_dataset`**.
This instance has an 80 GB disk (~13 GB free), so RF would fill the disk and crash, wasting the
~$7/10h. A disk-trimmed fresh pool can't reach the ≤2–4 visual-epoch "diversity" regime either
(≥2.75M unique images ≈ ~290 GB). G0.2 (streamed smoke) validates the fresh loader *parses* real
data, but does NOT exercise the full-load disk path.

**To run RF, pick one (Mac/director decision):**
1. Add a **streaming training path** (HF `IterableDataset` + shuffle buffer) so fresh data is
   consumed on-the-fly without materializing 1.49 TB. Smallest change; keeps this hardware.
2. Provision a **multi-TB / volume-backed** instance and run the existing non-streaming loader.
3. Use a smaller already-local fresh shard only for a *lower-token* pilot (loses the ≤2-epoch
   property — not a clean Gate-A retest).

Everything else this session (narrow Gate A at 1B, G0.2 loader parse-check, R2 Gate-B seeds,
R3 reprobe) completed and is on HF + git.
MD
log "RF HARD-SKIPPED (1.49TB pool vs 80GB disk) -- see RF-BLOCKED.md (G0.2 FRESH_OK=$FRESH_OK)"
commit "S3 RF: hard-skipped fresh-1B on this instance (1.49TB pool vs 80GB disk); blocker documented"

############ R3 — reprobe Session-1 ckpts (still local), then free their disk ############
for d in baseline g1gate sigmoid textinit; do
  [ -d "runs/$d" ] || continue
  reprobe_all "$d" "runs/$d"; U=$(upload_bg "runs/$d/reprobe"); commit "S3 R3: reprobe S1 $d"
  wait "$U" 2>/dev/null
done

python -u gate_summary.py >gate_summary.txt 2>&1 || log "gate_summary failed (non-fatal)"
cat gate_summary.txt; commit "S3 acceptance-gate summary (Gate A narrow + Gate B seeds)"
log "ALL SESSION-3 WORK COMPLETE -> hard_stop via EXIT trap"
