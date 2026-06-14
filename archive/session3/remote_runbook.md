# REMOTE-RUNBOOK — Session 3 (fresh-data arm)

**Author:** Mac agent (M4). **For:** remote (vast.ai instance `40436103`, 1× RTX 4090 48GB).
**Branch:** `freshdata-arm`. **You train; the Mac only authored + smoke-tested.**

## Why this session
Run-1 shows the concentration sink still absent at 470M tokens, so token-count undertraining
is no longer the worry — but those tokens are repeats (147,755 unique images ≈ 40 visual
epochs at ~1B; train loss collapsing while val rises = memorization). "Softmax VLM forms no
concentration sink" is now blocked by a **data-repetition confound**, not token count. The
decisive test is **more unique data, not more tokens**: a fresh-data baseline where 1B tokens
≈ 1–2 epochs of *fresh* images.

## Data decision (READ — the directive's premise was infeasible)
The entire **the_cauldron is only ~1.88M unique images** (47 subsets), so "more cauldron
subsets" cannot reach ≥3M / ≤2 epochs at 1B. Fresh data therefore uses **FineVision** (same
`{images, texts:[{user,assistant}]}` schema → drop-in through the existing collator/template;
no template change, invariant preserved). Fresh pool = 11 natural-image configs (`data/mixes.py:FRESH`):

`objects365_qa, densefusion_1m, allava_laion, lnqa, google_landmarks, lvis_instruct4v,
localized_narratives, LLaVA_Instruct_150K, coco_colors, image_textualization(filtered),
sharegpt4v(coco)`

- **Unique images: 4,644,331** (computed offline; loader re-asserts at init from `len(ds)`).
- **Projected visual-epochs @ 1.0B tokens: 1.68 (nominal 128 tok/sample) / 2.39 (effective
  ~90).** The loader hard-fails fresh mode above 4 epochs and warns above 2. 2.39 is in the
  warn band — a ~17× repetition reduction vs the repeated baseline's ~75 (eff.) epochs.
  Note this caveat in the report; do NOT headline it.
- Synthetic / document / chart / OCR / math / text-only FineVision configs are excluded to keep
  the fresh distribution close to the COCO-heavy repeated baseline (isolates *repetition*, not
  domain shift). Some COCO overlap with the repeated set remains (<3% of the fresh pool).

## Ordered runs (one launcher: `run_session3.sh`)
Token targets use `--max_tokens_M` (effective accounting: 49 image + real text tokens).

1. **Finish Run-1 baseline (repeated) → 1.0B.** RESUME from the latest `runs/baseline_ext`
   checkpoint (do not recompute). Set `R1_CKPT/R1_STEP/R1_TOK` from the newest HF ckpt first.
   ```bash
   R1_CKPT=runs/baseline_ext/ckpt_step<N>.pt R1_STEP=<N> R1_TOK=<tokens> bash run_session3.sh
   ```
2. **Run-2 (Gate B):** seed-1 × {baseline, textinit, g1gate, sigmoid}, `--data_mode repeated`,
   → 100M each. Gate B = do the sigmoid/G1 arms separate from baseline by ~2B? (here re-checked
   at the cheaper 100M seed-1 point for a second seed).
3. **Run-3 — NEW fresh-data baseline (Gate-A retest):** `--data_mode fresh`, seed-0, → 1.0B.
   The headline test: does a concentration sink appear once images stop repeating?
   ```bash
   python train_sinks.py --arm baseline --data_mode fresh --seed 0 --batch_size 128 --compile \
     --max_tokens_M 1000 --out_dir runs/baseline_fresh
   ```
4. **Run-4:** reprobe saved ckpts (per-head v/h, raw sigmoid mass, image-swap) — seconds of GPU.

`run_session3.sh` chains all four, uploads ckpts + `probes.jsonl` to HF `nemesismaniac/vlm-sink-emergence-ckpts`
after each run, commits/pushes jsonl after each stage, and stops the instance at the end.
**Dry-run first:** `bash run_session3.sh --dry-run` prints every train/stop/shutdown call.

## Operational rules (baked into the launcher)
- **Redundant auto-stop** (idle tail cost ~$1.5 last session): mechanism 1 = `stop_verify.sh`
  (`vastai stop`, polled until `actual_status` confirms); mechanism 2 = in-instance
  `shutdown -h +30`. Fired from an `EXIT` trap so it runs on clean finish OR crash. `watchdog.sh`
  is the independent backstop. **Never destroy `40436103` — start/stop only.**
- `WANDB_MODE=offline` exported.
- HF push after EACH run; local `*.pt` freed only after the upload lands.
- **rsync reminder:** `rsync -avz -e "ssh -p <port>" runs/ root@<ip>:/workspace/vlm-sink-emergence/runs/`
  to pull anything not on HF before stopping. `vastai start/stop instance 40436103`.
- **Wallet/time:** session cap ~$20–25; full budget $200, ~$7.5 spent. 4090 ≈ $0.70/hr →
  Run-1+Run-3 (two ~1B runs) ≈ 6–10 GPU-h, Run-2 ≈ 1–2 GPU-h. Stay under the cap; the watchdog
  has a 14h absolute deadline.

## Invariants (do NOT regress — carried hard-won rules)
- Matched template / seq-layout (49 img + 79 text, pos0 = first image token, no BOS) / optimizer
  / LR across repeated- and fresh-data baselines. **Data is the ONLY allowed difference.**
- Label-mask on vision tokens unchanged (loss-floor bug if regressed).
- Gate runs: `python3 tests/test_invariants.py` (0.5×-at-init equivalence) must pass first.
- Per-(layer,head) everywhere; direct ‖v‖ (not the α·‖v‖ proxy). The live probe now emits
  per-head v_ratio + raw (un-normalized) pos0 mass natively; `reprobe.py` stays the un-pooled
  product of record.
- Probe set is fixed/versioned (`mixes.PROBE_VERSION = v1-repeatedtail-32`) and identical across
  ALL arms/runs incl. Session-1 — comparability is via the reprobe products on this shared set.

## New flags on `train_sinks.py`
`--data_mode {repeated,fresh}` · `--val_size` (held-out images, primary val) ·
`--tok_per_sample` (epoch projection) · `--limit_per_subset N` (stream N rows/subset; smoke) ·
`--fake_data` (offline synthetic; smoke) · `--device {auto,cuda,mps,cpu}`. Each `--val_every`
logs **two** losses: `val_unseen` (primary, held-out images) and `val_seen` (memorization proxy).
