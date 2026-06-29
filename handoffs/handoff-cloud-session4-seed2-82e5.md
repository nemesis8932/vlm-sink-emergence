TO: Cloud Agent   FROM: Engineering Manager   RE: Session-4 — two seed-2 reruns (close Gate B at n=3), then exit-runbook to STOP (no destroy)

CONTEXT (rest in repo — Director `handoffs/handoff-em-session4-exit.md`, `docs/exit-runbook.md`, `docs/cloud-agent.md`):
- Gate A CLOSED (RF CONFIRM). Gate B OPEN at n=2. Only g1gate + textinit lack n=3 — baseline + sigmoid are solid, **do NOT rerun them**.
- REPEATED-data / map-style (the_cauldron {vqav2,cocoqa,aokvqa,vsr}) — NO streaming, NO leak, NO litdata. Run on the CURRENT box as-is.
- Recipe = byte-identical to the existing `runs/g1gate_seed1` / `runs/textinit_seed1` (read their `run_config.json`); ONLY change = `--seed 2`.

ASK (GPU-priority first, then exit):
1. **g1gate seed-2**: `--arm g1gate --seed 2 --vit_init pretrained --max_steps 100000 --batch_size 128 --grad_accum 1 --lr_lm 4e-4 --lr_mp 2e-3 --lr_vit 1e-4 --weight_decay 0.1 --probe_every 100 --probe_n 32 --val_every 500 --ckpt_steps 0,250,1000,2000,4000,6000` (repeated data, ~100M tok = matches seed-1). Signal to capture: **Sink^0.2_1** — is the near-zero (s0 0.004 / s1 0.011) reproducible at n=3?
2. **textinit seed-2**: same flags, `--arm textinit`. **Run to ~100M like seed-1** (NOT 60M — the directive's "~60M" is the seed-0 compare-floor snapshot; seed-1 actually ran to ~99.9M; matching it is the byte-identical choice + gives the 60M point for free). Signal: **h_ratio** — converges toward 42.5 (s0) or 5.5 (s1)? (concentration logged free.)
   Persist after EACH: commit `probes.jsonl`/`train_log.jsonl`/`run_config.json` to git + upload to HF `nemesismaniac/vlm-sink-emergence-ckpts`. A crash mid-arm must not lose the finished one.
3. **Then EXIT** per `docs/exit-runbook.md` steps 2→6 (step-1 generate.py comes from local-agent — wait for it on `sink-emergence`):
   - quality spot-check all arms on ~5 fixed held-out images → markdown (RF baseline kept regardless);
   - export keepers via `save_pretrained` → HF (finite quota: keepers only, final ckpt per arm, public repo if quota hit, mirror bulk to GDrive);
   - box-only sweep (scripts/watchdog/wandb/uncommitted edits → git; data/ckpts → HF); `git status` clean;
   - verify-elsewhere (fresh clone on another machine loads + generates each keeper; RF data fully on git+HF);
   - `vastai stop` → **independently confirm `actual_status=stopped`** (Session-1 lost ~$1.5 to a silent no-op). **STOP THERE — do NOT destroy; the user destroys the box manually.**

CONSTRAINTS: billed GPU; GPU busy back-to-back; persist-before-free; self-stop on waste. Director holds go/no-go; destroy is user-only.

RETURN (compacted, no raw logs): g1gate-s2 Sink^0.2_1 + textinit-s2 h_ratio (+concentration) vs the two existing seeds; HF/git paths; arm quality-check md; load-verified-elsewhere confirmation; `actual_status=stopped` confirmed + final compute log. Hand back to user for destroy.
