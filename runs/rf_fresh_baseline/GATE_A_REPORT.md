# Gate A — RF Fresh Baseline @ 1B tokens

**Run:** `baseline` arm, ViT pretrained, FineVision **fresh** stream, bs=128, seed=0.
Stopped deliberately at **1B tokens** (step 69775, tok 1,001,238,892). UTC 2026-06-15.

## Verdict: Gate A CONFIRMED (concentration sink absent, free of repeated-data confound)

**Sink^0.3_1 = 0.000 across the ENTIRE 0→1B run** (max over run = 0.000). The
attention-concentration sink does **not** emerge in the fresh baseline. Retested free
of the repeated-data overfitting confound → confirms the Session-1 Gate A finding.

## 3-signature summary @ 1B (probe step 69700, tok 1000.2M; probe v1-repeatedtail-32)

| signature | @ start (~57M) | @ 1B | net |
|---|---|---|---|
| **Sink^0.3_1** (attn concentration) | 0.000 | **0.000** | flat zero |
| max_a0 (max attn→pos0) | 0.131 | 0.098 | ~flat, ≪ sink |
| **v_ratio** (pos0 value-norm) | 0.787 | **0.692** | ↓ declines, plateaus |
| **h_ratio** (massive activation) | 2.40 | **3.22** | ↑ rises ~34%, plateaus |

**Decoupling result:** value-norm drops and massive-activation rises (h_ratio 1.4→3.2
over the run) **without any attention-concentration sink forming**. The value-norm /
massive-activation signatures move independently of attention concentration — the
hypothesised decoupling, now confirmed on fresh data at 1B.

## Training health
- loss 7.7 → ~0.4–0.8 (healthy)
- **fresh val_loss (unseen) 1.46 → 0.638**, tracks train, **no overfit gap** (expected on fresh data)

## Data-cleanliness gaps to assess (flagged)
1. **Resume discontinuity @ step 4000 (~57M tok):** run was resumed from `ckpt_step4000`
   after a dataloader-config change; resume = **weights-only (optimizer state reset)** →
   brief loss bump, recovered in a few hundred steps. Probe/loss continuity across the
   seam is otherwise smooth.
2. **Trajectory split:** current `train_log.jsonl`/`probes.jsonl` + `stdout_full.log`
   cover the resumed run (57M→1B). The 0→57M portion is in `*.pre_resume4000` files
   (earlier run, same seed/config; Sink^0.3_1 also 0.000 there).
3. **No exact-1B checkpoint:** nearest weights ckpt is `ckpt_step64000` (~915M tok);
   ckpt_steps lacked 70000. Dense probes (every 100 steps) fully cover 1B.
4. **Coarse streaming shuffle:** `DATA_SHUFFLE_BUFFER=500` (vs non-streaming repeated
   comparator's full shuffle) — minor shuffle-quality asymmetry, unique-image set intact.
5. **Effective visual epochs @1B = 2.39** (>2.0 ideal, within acceptable bound per script).

## Artifacts
- **git** (`b441a77`, pushed): `runs/rf_fresh_baseline/{train_log.jsonl, probes.jsonl,
  run_config.json, stdout_full.log, *.pre_resume4000}`
- **HF** `nemesismaniac/vlm-sink-emergence-ckpts` → `runs/rf_fresh_baseline/`:
  ckpt_step{0,250,1000,2000,4000,8000,16000,32000,64000}.pt
- `stdout_full.log` is the authoritative full-fidelity probe/loss log.

## Config (final)
workers=8, DATA_SHUFFLE_BUFFER=500, MALLOC_ARENA_MAX=2, prefetch=2, lr_lm 4e-4 /
lr_mp 2e-3 / lr_vit 1e-4, wd 0.1, compile. ~0.45 s/step on RTX 4090.
