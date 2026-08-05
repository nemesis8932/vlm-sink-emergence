# Gate A — RF Fresh Baseline @ 1B tokens

**Run:** `baseline` arm, ViT pretrained, FineVision **fresh** stream, bs=128, seed=0.
Stopped deliberately at **1B tokens** (step 69775, tok 1,001,238,892). UTC 2026-06-15.

## Verdict: Gate A CONFIRMED (concentration sink absent, free of repeated-data confound)

**Sink^0.3_1 = 0.000 across the ENTIRE 0→1B run** (max over run = 0.000). The
attention-concentration sink does **not** emerge in the fresh baseline. Retested free
of the repeated-data overfitting confound → confirms the Session-1 Gate A finding.

## 3-signature summary, init→1B (probe step 69700, tok 1000.2M; probe v1-repeatedtail-32)

Anchored at **init (step 0)**, single continuous trajectory across the step-4000 resume
seam (Auditor-verified continuous — see "resume" gap below). The earlier table anchored
at ~57M (the seam value), which understated the rise; corrected here.

| signature | @ init (step 0) | @ 1B | net |
|---|---|---|---|
| **Sink^0.3_1** (attn concentration) | 0.000 | **0.000** | flat zero, entire run |
| max_a0 (max attn→pos0) | 0.056 | 0.098 | ~flat, ≪ sink threshold |
| **h_ratio** (massive activation) | 1.43 | **3.22** | ↑ +130%, rise continues post-warmup |
| v_ratio (pos0 value-norm) | 1.00 | 0.69 | ↓ net, non-monotone (dips 0.66 @16k, recovers) |

**Decoupling result (Auditor ruling):** the headline dissociation is **h_ratio rises
while Sink^0.3_1 stays pinned at zero**. h_ratio's rise is *not* a warmup artifact — it
continues monotonically post-seam (2.40→3.22 over 57M→1B). v_ratio is supporting context:
it ends below init but ~75% of its drop is the 0–57M warmup and it partially recovers
after, so it is **not** framed as emergence. Massive-activation grows across training
without any attention-concentration sink forming — the hypothesised decoupling, confirmed
on fresh data at 1B.

## Training health
- loss 7.7 → ~0.4–0.8 (healthy)
- **fresh val_loss (unseen) 1.46 → 0.638**, tracks train, **no overfit gap** (expected on fresh data)

## Data-cleanliness gaps — Auditor rulings (2026-06-15)
1. **Resume @ step 4000 (~57M tok), optimizer reset → RULED OUT for decoupling.** Resume
   was **OOM crash-recovery** (proc `Killed`; config retuned for memory), not a data fix —
   same fresh pool both sides. Reset = weights-only (Adam state discarded), bumped *loss*
   but **not** the v/h *signatures*: (a) the steep decoupling move completes by step 4000,
   *before* the seam; (b) steps 4000–4600 are double-covered (pre_resume to 4600, main from
   4000) — at the shared ckpt pre/main are identical (vr 0.787/0.787, hr 2.395/2.395) and
   over the overlap they diverge ≤ probe noise with no transient; sink=0.000 both branches.
   Breaks byte-identical-to-Comparator rule → matters for Gate-A *cross-run* comparability,
   not for the *within-RF* decoupling trajectory.
2. **Trajectory split (cosmetic).** `*.pre_resume4000` = 0→57M, main = 57M→1B; stitched
   continuous in `decoupling_stitched.json`. Sink^0.3_1 = 0.000 in both segments.
3. **No exact-1B checkpoint (cosmetic).** Nearest weights ckpt `ckpt_step64000` (~915M);
   dense probes (every 100 steps) fully cover to 1B. Verdict reads off probes, not ckpt.
4. **Streaming shuffle `DATA_SHUFFLE_BUFFER`: 1500 (pre) → 500 (main) → RULED OUT,
   strengthened.** Both changes OOM-driven memory tuning (per git history of run script),
   same unique-image pool. Decoupling + sink=0.000 appear in **both** the stronger-shuffle
   (1500) and weaker-shuffle (500) regimes → weak shuffle did **not** manufacture the
   no-sink result. Caveat-level, document; no rerun.
5. **Effective visual epochs @1B = 2.39** (>2.0 ideal, within script bound). Caveat-level.

## Artifacts
- **git** (`b441a77`, pushed): `runs/rf_fresh_baseline/{train_log.jsonl, probes.jsonl,
  run_config.json, stdout_full.log, *.pre_resume4000}`
- **HF** `nemesismaniac/vlm-sink-emergence-ckpts` → `runs/rf_fresh_baseline/`:
  ckpt_step{0,250,1000,2000,4000,8000,16000,32000,64000}.pt
- `stdout_full.log` is the authoritative full-fidelity probe/loss log.

## Config (final)
workers=8, DATA_SHUFFLE_BUFFER=500, MALLOC_ARENA_MAX=2, prefetch=2, lr_lm 4e-4 /
lr_mp 2e-3 / lr_vit 1e-4, wd 0.1, compile. ~0.45 s/step on RTX 4090.
