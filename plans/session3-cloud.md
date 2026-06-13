# SESSION 3 — Cloud runbook (authoritative)

**Instance:** vast.ai 40436103 (RTX 4090, ~$0.70/hr). **Scope:** finish Run-1 + Run-2 seeds +
fresh-data baseline (1B) + Run-3 reprobe. **Branch:** `freshdata-arm` (run off it; do NOT merge
until the fresh-1B validates the loader at scale). **Script:** the Mac agent's `run_session3.sh`
(`--dry-run` verified) — this doc is the order + go/no-go layer around it.
**Session wallet cap: $25** (expected ~$14). Project spend to date ≈ $7.5 / $200.

> **State at launch:** Run-1 (baseline extend, repeated data) is LIVE at 770M/1000M. It finishes
> in place (~2h). Do NOT relaunch or resume it — the `R1_CKPT/R1_STEP/R1_TOK` params are moot
> this session. Everything new waits until Run-1 frees the GPU (single instance, sequential).

---

## §0 — Pre-flight gates (must pass IN ORDER before any 1B spend)

- [ ] **G0.1 — Let Run-1 finish to 1.0B.** On completion, capture the final REPEATED-data baseline:
      `Sink^{0.2/0.3/0.4}₁`, first-crossing token (expect NONE), `mean/max attn→pos0`, `v_ratio`,
      `h_ratio`, and `val_unseen` vs `val_seen`. Record as the Gate-A-narrow result
      (*"concentration absent at 1B / ~75 epochs of repeated data; norm signatures stable"*).
- [ ] **G0.2 — Real-data fresh smoke (HARD GATE, ~$0.15).** The `fresh` loader was only tested with
      `--fake_data` on the Mac (torchvision blocked). Run ~200 real steps on FineVision:
      verify the 11 configs download + parse (schema `{images, texts:[{user,assistant}]}`),
      probe validates eager-fp32-vs-forward (<1e-2), image-swap self-check returns "varies",
      `val_unseen`/`val_seen` both log. **If this fails, fix the loader — do NOT launch fresh-1B.**
- [ ] **G0.3 — Confirm config & guards.** `HF_CKPT_REPO = nemesismaniac/vlm-sink-emergence-ckpts`;
      probe set `v1-repeatedtail-32` held out from the FineVision pool too (not just the repeated
      set); redundant auto-stop armed (`stop_verify.sh` + `shutdown -h +30` EXIT trap);
      `WANDB_MODE=offline`; all runs in tmux; wallet alarm at $25.

---

## §1 — Run order (sequential on 40436103)

| # | Run | data_mode | tokens | seeds | ~hrs | ~$ | answers |
|---|---|---|---|---|---|---|---|
| R2 | seed-2 × 4 arms | repeated | 100M each | seed-1 | ~3.8 | ~2.7 | **Gate B** (do arms separate across seeds?) |
| RF | **fresh-data baseline** | **fresh** | **1.0B** | seed-0 | ~10 | ~7 | **Gate A retest** (does concentration form on diverse data?) |
| R3 | reprobe saved ckpts | — | — | — | <1 | ~0.5 | per-head un-pooled v/h, raw sigmoid mass, image-swap (closes FAIL-B holes) |

Order rationale: R2 is cheap and feeds Gate B; RF is the decisive long run; R3 needs R2+RF ckpts,
so it's last. (Optional: to halve wall-clock, RF can run on a SECOND cheap instance in parallel
with R2 — only if you'll babysit two auto-stops. Default: sequential, simpler.)

---

## §2 — Acceptance / go-no-go per run

### Gate B (R2 — seed agreement at matched 100M)
For each arm, compare seed-0 (S1) vs seed-1 on `Sink^0.2₁`, `mean attn→pos0`, `v_ratio`, `h_ratio`.
- **sigmoid** (0.83 vs 0.00 effect) — expected to survive almost any noise. Confirm.
- **g1gate** (~0.004, volatile) — THE fragile arm. If seeds disagree materially → flag for a
  3rd seed, and do NOT let "G1 creates a sink" near an abstract until stable.
- Honest limit: n=2 can't give true 2σ. Practical bar = tight agreement + large effect. Note it.

### Gate A retest (RF — the decision the session exists for)
Compare fresh (≈2.39 epochs) vs repeated (≈75 epochs) baseline concentration:
- **Concentration forms on fresh but not repeated** (heads cross 0.2 / `Sink^0.3₁` ≥ ~5%) →
  **diversity-driven sink** → the stronger paper (*"multimodal concentration sinks are driven by
  data diversity, not token count"*). Pivot the thesis toward this.
- **Still 0 on fresh** → the absence is real, not a repetition artifact → strong claim, report
  with the 2.39-epoch caveat (do NOT headline as "no sink ever"; it's "not observed at ≤2.4
  epochs of fresh natural-image data").
- Either branch is reportable (heads-I-win). Also confirm `val_unseen` overfits far less than the
  repeated run — that corroborates the repetition story regardless of the sink outcome.

---

## §3 — Shutdown
1. HF push ckpts + `probes.jsonl` after EACH run (not just at end).
2. `rsync -avz <instance>:/workspace/runs/ ./runs/` — needed for Mac analysis/figures.
3. `vastai stop instance 40436103` — **verify it stopped** (it didn't last session). Never destroy.

## §4 — Cost ledger (this session)
Run-1 tail ~$1.5 · G0.2 smoke ~$0.15 · R2 ~$2.7 · RF ~$7 · R3 ~$0.5 · buffer ~$2 → **≈ $14**.
Project running total ≈ **$21 / $200**. Plenty of runway; the cap is to catch a stuck instance,
not because money is tight.

## §5 — Abort conditions
- G0.2 smoke fails → stop, fix loader on the Mac, re-stage. No 1B spend on a broken loader.
- Instance won't auto-stop → manual `vastai stop`; do not start the next run until confirmed idle-safe.
- Any arm's probe image-swap check returns "pos0 invariant" → harness bug or artifact; halt that
  arm and investigate before trusting its numbers.

## §6 — Decision point AFTER this session (for the director)
Green-light the **Stage-2 ~1B A100 confirmation (~$45–70)** only if BOTH hold:
1. Gate B passes (dissociation separates across seeds at matched budget), AND
2. Gate A resolves cleanly (fresh-baseline gives a clear forms / doesn't-form answer).
Otherwise: add seed-3 for the fragile arm, or ship the 222M sweep as a focused workshop paper.

## PR
Open the `freshdata-arm` PR for the diff record. **Do not merge** until RF (fresh-1B) confirms the
loader holds at scale; merge after.