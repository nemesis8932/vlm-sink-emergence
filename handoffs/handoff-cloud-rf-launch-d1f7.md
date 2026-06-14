TO: Cloud Agent   FROM: Engineering Manager   RE: wire infra, run pre-launch gates, launch RF overnight

CONTEXT (rest in repo — Director brief `handoffs/handoff-em-rf-baseline.md`, `docs/cloud-agent.md`, `docs/adr/0002-*`):
- Mac delivers loader + `compute_log.jsonl` monitor + failsafe env-var profile on `sink-emergence` BEFORE you launch. Your job = integrate + validate-on-real-data + launch, not rebuild.
- Recipe LOCKED (ADR-0002): batch 128, grad_accum 1, comparator LR/max_steps. Do NOT grow batch to fill VRAM — compute is 100%, zero gain, breaks the Comparator clone.

ASK (gates are HARD — STOP, don't auto-advance):
1. **Comparator config — RESOLVED off-box (EM verified both `runs/baseline` + `runs/baseline_ext` run_config.json).** Comparator = ONE 100k-step cosine, executed in two segments (baseline 0→18287 hit a 1.6h cap, baseline_ext resumed→100000); both share `max_steps=100000`. RF clones it as a SINGLE un-segmented run, NO resume. Clean swap — not a STOP.
   **RF recipe (locked):** `--arm baseline --vit_init pretrained --data_mode fresh --max_steps 100000 --batch_size 128 --grad_accum 1 --seed 0 --lr_lm 4e-4 --lr_mp 2e-3 --lr_vit 1e-4 --weight_decay 0.1 --compile` ; `lm_max_length 79`. NOTE: `max_steps=100000` (~954M tok @ ~9.5k tok/step), NOT 105k — match STEPS not the rounded "1B". Only intended diff vs comparator: data = `mixes.FRESH`, not `vqav2,cocoqa,aokvqa,vsr`.
2. **Disk dry-run** (the seam Mac left): short real streamed run; confirm async shard-eviction keeps cache bounded on the 80GB box vs ~1.49TB pool. Over-evict→re-download (fine), never crash. Confirm bounded BEFORE the multi-hour launch.
3. **Wire monitor + failsafe** into `run_session`: export `NAN_KILL=1 VAL_KILL=0 CAP_HOURS=8`; verify watchdog reads them; smoke-test that `stop_verify.sh` ACTUALLY stops (last session lost ~$1.5 to a stop that didn't take — verify `actual_status`).
   **HF token was rotated** — if HF push/pull 401s (old token on box), STOP and ask the user for the new token (`huggingface-cli login --token …` / set `HF_TOKEN`); do not burn the run on a dead credential.
4. **Launch RF**: `--arm baseline` (pretrained ViT, random LM), `--data_mode fresh`, 1B tok, single seed. Probe every 100 to ~5k then every 500. bf16 ckpts at 0,250,1k,2k,4k,8k,16k,32k,64k,105k → HF. Overnight = checkpoint-only, NO divergence/val auto-kill (Director judges AM); crash/NaN/8h-cap backstop stays.
5. **Mid-run bug**: you may self-fix, but MUST commit+push the fix to `sink-emergence` (box is wiped on stop), then stop→fix→resume from last bf16 ckpt — relaunch needs Director go/no-go.

CONSTRAINTS: billed GPU ~5–6h; GPU busy back-to-back; persist-before-free; self-stop on waste. Surface every go/no-go to EM→Director.

RETURN (compacted, no raw logs): pre-launch — disk bounded ✓ + comparator config found (or STOP); post-run — Sink^0.3_1 trajectory+final, fresh val_loss trajectory (overfit y/n), 3-signature summary @1B, HF paths, GPU-hrs×$/hr spend from the cost ticker. Gate-A verdict: Sink^0.3_1 <~5% AND val healthy = clean CONFIRM.
