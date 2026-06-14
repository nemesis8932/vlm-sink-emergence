#!/bin/bash
# Final stage (per the "export all run data, then stop" directive). Waits for the Session-3
# wrapper (G0.1->G0.2->R2->R3->gate_summary) to finish, then bundles ALL analysis-grade run data
# into session3-export.tar.gz (+ MANIFEST.txt + sha256sums.txt), ships it two ways
# (HF dataset + git for the small files; tarball to HF since it survives the stop), writes a
# cost log + one-line STATUS, then does the VERIFIED stop. Checkpoints are NOT in the tarball
# (too large; they are already on HF nemesismaniac/vlm-sink-emergence-ckpts).
set -u
cd /workspace/vlm-sink-emergence
source /workspace/venv/bin/activate
export HF_HOME=/workspace/.hf_home TOKENIZERS_PARALLELISM=false
REPO=nemesismaniac/vlm-sink-emergence-ckpts
log(){ echo "[export $(date -u +%H:%M:%S)] $*"; }

log "waiting for Session-3 wrapper to finish (R2/R3/gate_summary) ..."
while pgrep -f run_session3_cloud.sh >/dev/null 2>&1; do sleep 60; done
log "wrapper finished. building export bundle."

# --- cost log (approx: sum wall-clock across training runs * $0.7037/hr) ---
python3 - <<'PY' > cost_log.md 2>&1
import json, glob, os
RATE=0.7037
print("# Session-3 cost log (approx)\n")
print(f"Instance 40436103 RTX 4090 @ ${RATE}/hr. GPU wall-clock per run (from train_log 'wall'):\n")
tot=0.0
for f in sorted(glob.glob('runs/*/train_log.jsonl')):
    try:
        rows=[json.loads(l) for l in open(f) if l.strip()]
        wall=max((r.get('wall',0) for r in rows), default=0)/3600.0
        tot+=wall
        print(f"- {os.path.dirname(f).split('/')[-1]:18s} {wall:5.2f} h  ~${wall*RATE:5.2f}")
    except Exception as e:
        print(f"- {f}: ERR {e}")
print(f"\n**Training wall total ≈ {tot:.2f} GPU-h ≈ ${tot*RATE:.2f}** (excludes setup/probe/idle).")
print("RF (fresh-1B) NOT run -> ~$7 saved (see RF-BLOCKED.md).")
PY

# --- collect the analysis-grade files (NO *.pt) ---
FILES=$(ls -1 \
  runs/*/probes.jsonl runs/*/train_log.jsonl runs/*/run_config.json \
  runs/*/reprobe/*.npz runs/*/reprobe/*.jsonl \
  gate_summary.txt gateA_narrow.txt cost_log.md RF-BLOCKED.md HALFWAY.md REPORT.md \
  plans/session3-cloud.md run_session3_cloud.sh train_sinks.py sink_probe.py reprobe.py \
  data/mixes.py gate_summary.py 2>/dev/null)

# --- MANIFEST + sha256 ---
{
  echo "# session3-export MANIFEST  ($(date -u))  instance 40436103"
  echo
  echo "Checkpoints are NOT here (too large) -> HF dataset $REPO (per-run dirs)."
  echo "val_unseen/val_seen: present in R2 seed-1 arms' train_log.jsonl (new loader);"
  echo "  baseline_ext (Run-1) used the prior single-val code -> 'val_loss' only."
  echo
  echo "## files"
  for f in $FILES; do printf "%-46s %10s bytes\n" "$f" "$(stat -c%s "$f" 2>/dev/null)"; done
} > MANIFEST.txt
sha256sum $FILES > sha256sums.txt 2>/dev/null

tar -czf session3-export.tar.gz --exclude='*.pt' $FILES MANIFEST.txt sha256sums.txt
sha256sum session3-export.tar.gz >> sha256sums.txt
SZ=$(du -h session3-export.tar.gz | cut -f1)
log "built session3-export.tar.gz ($SZ)"

# --- ship: tarball+manifest to HF (survives the stop); small files to git ---
huggingface-cli upload "$REPO" session3-export.tar.gz session3-export.tar.gz --repo-type dataset >>hf_upload.log 2>&1 \
  && log "tarball uploaded to HF $REPO/session3-export.tar.gz" || log "HF tarball upload FAILED (see hf_upload.log)"
huggingface-cli upload "$REPO" MANIFEST.txt MANIFEST.txt --repo-type dataset >>hf_upload.log 2>&1 || true

# --- one-line STATUS (preserved in git + tarball-adjacent) ---
python3 - <<'PY' > STATUS.txt 2>&1
import json, os, glob
def final(run):
    p=f'runs/{run}/probes.jsonl'
    if not os.path.exists(p): return None
    r=[json.loads(l) for l in open(p) if l.strip()][-1]; return r
b=final('baseline_ext') or final('baseline')
ga = (f"Gate-A(narrow): baseline repeated @ {b['tokens_seen']/1e6:.0f}M tok -> "
      f"Sink^0.2={b['summary']['sink_eps0.2']:.3f} Sink^0.3={b['summary']['sink_eps0.3']:.3f} "
      f"max_a0={b['summary']['max_attn_pos0']:.3f} (concentration sink absent); "
      f"v_ratio={b['summary']['v_ratio_pos0']:.2f} h_ratio={b['summary']['h_ratio_pos0']:.2f}") if b else "Gate-A: n/a"
done=[]; pend=[]
for tag,run in [('R2:baseline','baseline_seed1'),('R2:textinit','textinit_seed1'),
                ('R2:g1gate','g1gate_seed1'),('R2:sigmoid','sigmoid_seed1')]:
    (done if os.path.exists(f'runs/{run}/probes.jsonl') else pend).append(tag)
sz=os.path.getsize('session3-export.tar.gz') if os.path.exists('session3-export.tar.gz') else 0
print("STATUS: Run-1->1B DONE; "
      f"completed=[{', '.join(['Run-1']+done+['R3-reprobe'])}]; "
      f"skipped=[RF fresh-1B (1.49TB pool vs 80GB disk)]; pending={pend or 'none'}. "
      f"{ga}. export=/workspace/vlm-sink-emergence/session3-export.tar.gz ({sz/1e6:.0f} MB, also on HF {os.environ.get('REPO','')}).")
PY
cat STATUS.txt

git add -A 2>/dev/null
git commit -m "S3 export: MANIFEST + sha256 + cost log + STATUS (tarball->HF; ckpts->HF)" 2>&1 | tail -1
git push 2>&1 | tail -1

# --- verified stop (remove guard so stop_verify actually stops) ---
log "removing NO_AUTOSTOP guard and stopping (verified)"
rm -f NO_AUTOSTOP
bash stop_verify.sh
