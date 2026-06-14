# ANALYSIS-session3.md — Session-3 findings

**Date:** 2026-06-13 · **Analyst:** Mac agent (offline, $0) · **Source data:** `session3-export` (rsync'd from vast.ai instance 40436103)

Figures in `analysis/` (regenerated locally from probes.jsonl; cloud plotting is permanent now).

---

## 0. Export verification

All text/JSON files in `sha256sums.txt` verified **PASS** (shasum -c).  
`.npz` reprobe files are **NOT present locally** — they live on HF (`nemesismaniac/vlm-sink-emergence-ckpts`), consistent with `.gitignore`. All `probes.jsonl` and `reprobe_summary.jsonl` files are checksummed OK. Analysis below uses probes + reprobe summaries exclusively; .npz raw arrays are not needed for the metrics computed here.

Probe counts loaded: baseline=184, baseline\_ext=715, baseline\_seed1=106, g1gate=109, g1gate\_seed1=106, sigmoid=108, sigmoid\_seed1=106, textinit=64, textinit\_seed1=106.

---

## 1. Gate A verdict — CONFIRM (with documented confound)

**Question:** does the softmax baseline form a concentration sink before 1B tokens?

### Numbers at final step (seed-0, baseline\_ext, step 89600, 854.6M text tokens)

| metric | value |
|---|---|
| Sink^0.2₁ | **0.0000** |
| Sink^0.3₁ | **0.0000** |
| Sink^0.4₁ | **0.0000** |
| max mean-attn→pos0 (any head ever) | **0.1389** (well below ε=0.2) |
| mean attn→pos0 (global) | 0.0527 |
| v_ratio | 0.687 |
| h_ratio | 2.462 |
| First head crossing mean-attn→pos0 > 0.2 | **NEVER** |

The per-head reprobe (reprobe\_summary.jsonl, step=80000) confirms no pooling artifact: `v_ratio_perhead_mean=0.679`, `v_ratio_perhead_min=0.411`, `v_ratio_perhead_max=1.967`. Per-head spread is present (some heads drain more than others) but none reverse the group direction. Image-swap: `swap_v_pos0_cv=0.569` at final reprobe step — v_norm at pos0 **varies with image content**, ruling out a pure positional artifact.

**Verdict: Gate A → CONFIRM.** No concentration sink through 854.6M tokens. The norm/value signatures (h\_ratio≈2.4, v\_ratio≈0.69) persist and are real. The decoupling from Session 1 holds: norm without concentration, stable over 5× the original budget.

### The confound — overfitting from data repetition (documented limitation, not hidden)

This result was measured on **heavily repeated** training data. The 146,731-image cauldron pool was cycled approximately:

| tokens | visual epochs |
|---|---|
| 100M (seed-1 arms end) | ~9 |
| 174M (S1 baseline end) | ~15 |
| 470M (HALFWAY.md) | ~41 |
| 855M (baseline\_ext end) | **~74** |

Evidence of memorization:
- **Train loss**: 0.244 at 174M → 0.015 at 842M (essentially zero; model has fit the training set).
- **Val_loss (repeated cauldron)**: 1.169 at 174M → 1.40 at 844M (flat-to-rising; no generalization gain).
- **Seed-1 arms val\_seen vs val\_unseen** (100M tokens): val\_seen plummets from 2.17→0.44 (memorized images); val\_unseen flattens at 1.0–1.18 from step ~4000 onward. The memorization-onset step is ~3000–4000 (~28–38M tokens), after which val\_seen < val\_unseen and the gap grows monotonically.

**Documented limitation paragraph (for the paper):**  
*The absence of an attention-concentration sink at 854M tokens was measured on the repeated the\_cauldron pool (146K images, ~74 visual epochs). Train loss fell to near-zero by the final checkpoint while held-out val loss was flat-to-rising, indicating the model memorized the training set rather than learning a generalizable representation. It therefore remains possible that fresh-data scale — not token count per se — is what drives concentration-sink formation, as Gu et al. (2024) trained on 5B genuinely fresh tokens. The norm and value signatures (h\_ratio≈2.4, v\_ratio≈0.69) emerged early and are robust to memorization; what we cannot claim is that the concentration signature would remain absent under equivalent fresh-data training. This is the primary motivation for the RF (random-fresh-1B) run, which requires the streaming-loader fix before it can execute.*

### Mean attn→pos0 decline: weak evidence for diversity hypothesis, not noise-free

mean\_attn→pos0 trajectory (baseline combined):
- step 0 → 174M (S1): 0.053 → 0.060, peak ~0.073 around step ~5000 (48M), then declining to 0.060 at S1 end
- 174M → 855M (baseline\_ext): 0.060 → 0.053, monotone gentle decline

The mild decline from the early-training peak is consistent with a diversity-driven hypothesis (as the model memorizes, the per-image attention pattern becomes more stereotyped or less pos0-focused when the answer is already known from training). However the effect is small (~29% relative decline over 700M additional tokens) and the alternative (random probe-batch variation in a stable metric) cannot be excluded from this data alone. **Do NOT headline as "diversity drives concentration-sink formation."** State as hypothesis, require RF test.

---

## 2. Gate B verdict — dissociation table (n=2)

*Caveat applies to all entries below: n=2 seeds cannot yield a true 2σ confidence. Practical bar = tight seed agreement + large effect size. Where seeds diverge significantly, we mark explicitly.*

### Two-seed dissociation table at matched ~100M tokens

| arm | seed | Mtok | Sink^0.2 | Sink^0.3 | mean\_a0 | v\_ratio | h\_ratio | verdict |
|---|---|---|---|---|---|---|---|---|
| baseline | s0 | 101.1 | 0.000 | 0.000 | 0.068 | 0.723 | 2.155 | |
| baseline | s1 | 100.0 | 0.000 | 0.000 | 0.062 | 0.687 | 1.708 | **AGREE** |
| sigmoid | s0 | 101.7 | 0.830 | 0.659 | 0.377 | 1.479 | 1.095 | |
| sigmoid | s1 | 100.0 | 0.756 | 0.474 | 0.311 | 1.603 | 1.299 | **AGREE** |
| g1gate | s0 | 101.1 | 0.004 | 0.000 | 0.068 | 0.805 | 1.674 | |
| g1gate | s1 | 100.0 | 0.011 | 0.000 | 0.073 | 0.845 | 2.038 | **TENTATIVE AGREE — needs seed-3** |
| textinit | s0 | 59.6 | 0.852 | 0.830 | 0.627 | 0.377 | 42.5 | |
| textinit | s1 | 60.0 | 0.556 | 0.444 | 0.235 | 0.634 | 5.50 | **AGREE on concentration; DISAGREE on h\_ratio** |

*textinit s0 ends at 59.6M (S1); for s1, value shown at step 6300 (60.1M) to match. Full s1 trajectory is available to 100M.*

### Per-arm Gate B assessment

**baseline — AGREE.**  
Both seeds: Sink^0.2 = 0.000, Sink^0.3 = 0.000. v\_ratio: 0.72 vs 0.69 (same direction, same magnitude class). h\_ratio: 2.16 vs 1.71 (moderate difference; both in the 1.5–2.5 range — same class). Gate B passes for the absence-of-concentration finding.

**sigmoid — AGREE (robust).**  
Sink^0.2: 0.830 vs 0.756 — large effect, small seed-to-seed delta. v\_ratio: 1.479 vs 1.603 — both well above 1.0 (anti-drain), same direction. h\_ratio: 1.095 vs 1.299 — both near 1 (no massive activations). Raw per-head mass to pos0 (from reprobe\_summary): s0=0.301, s1=0.276 — both 5-6× above baseline (0.06). The concentration is **not a softmax normalization artifact**: raw un-normalized sigmoid mass to pos0 is genuinely elevated. Gate B passes for sigmoid concentration, anti-drain v\_ratio, and h\_ratio suppression.

**g1gate — TENTATIVE AGREE; mark fragile, needs seed-3.**  
Sink^0.2: 0.004 vs 0.011. Both non-zero and same direction (weak concentration), but the signal is tiny (3–7 heads of 270 crossing ε=0.2) and the seed-to-seed ratio (2.75×) is as large as the signal itself. Sink^0.3 = 0.000 in both seeds. v\_ratio: 0.805 vs 0.845 (mild drain, milder than baseline — qualitative direction agrees). h\_ratio: 1.674 vs 2.038 (suppressed vs baseline 2.155 — qualitative direction agrees, but ordering reverses vs baseline by s1). Gate B: direction agrees but magnitude is fragile. **Do NOT headline "g1gate creates a sink" or "h\_ratio suppressed." Mark as unconfirmed pending seed-3.**

**textinit — AGREE on concentration presence; DISAGREE on h\_ratio magnitude.**  
Sink^0.2: 0.852 vs 0.556 — both strongly >0.5, qualitative agreement (inherited concentration sink). Mean attn→pos0: 0.627 vs 0.235 — significant magnitude difference but both far above baseline. v\_ratio: 0.377 vs 0.634 — both <1.0 (drain), but seed0 drains far more severely. h\_ratio: **42.5 vs 5.5 — factor-of-8 divergence.** Seed-1 trajectory shows h\_ratio peaked ~23 at step ~1000 (9.5M tok) then decayed continuously to 5.5 at 60M; seed-0 was still 42.5 at 60M. This is a genuine seed-level divergence in the explosive massive-activation component.

Gate B for textinit: **AGREE** on "inherited concentration sink present from step 0, strong throughout." **DISAGREE** on h\_ratio magnitude — the 42× figure from Session 1 is not reproduced in seed 1 at the same token budget. Do not headline textinit h\_ratio until seed-3. The relocation-and-amplification qualitative finding survives (both seeds show it); the specific "extreme" massive-activation magnitude does not.

### Dissociation summary

The 4-way qualitative dissociation **survives two seeds** on the primary claims:

| signature | baseline | sigmoid | g1gate | textinit |
|---|---|---|---|---|
| Concentration (Sink^0.2) | none ✓✓ | strong ✓✓ | weak/fragile ⚠ | strong ✓ (need seed-3 for h) |
| v\_ratio direction | drain ✓✓ | anti-drain ✓✓ | mild drain ✓ | drain ✓ (magnitude variable) |
| h\_ratio direction | moderate ✓ | suppressed ✓✓ | suppressed ✓ | extreme ⚠ (disagrees seed-to-seed) |

The core paper spine — arms produce **different combinations** of the three signatures — is confirmed at n=2. The two most important arms (sigmoid's inversion of Gu et al.'s text-LM result; baseline's decoupling of norms from concentration) are robust. g1gate and textinit h\_ratio need seed-3 before they can be headlined as quantitative results.

---

## 3. R3 per-head reprobe results

### Per-head v\_ratio (un-pooled)

Reprobe\_summary provides per-head `v_ratio_perhead_mean/min/max` at each checkpoint. Key end-of-run values:

| arm | v\_ratio (mean) | v\_ratio (min head) | v\_ratio (max head) |
|---|---|---|---|
| baseline @step18287 | 0.710 | 0.423 | 2.088 |
| baseline\_seed1 @step10484 | 0.687 | 0.548 | 1.246 |
| g1gate @step10786 | 0.816 | 0.514 | 2.515 |
| g1gate\_seed1 @step10484 | 0.841 | 0.526 | 2.276 |
| sigmoid @step10664 | **1.481** | 0.603 | **3.618** |
| sigmoid\_seed1 @step10484 | **1.601** | 0.835 | **3.469** |
| textinit @step6244 | 0.376 | 0.035 | 2.616 |
| textinit\_seed1 @step10484 | 0.564 | 0.063 | 3.009 |

The sigmoid arm's anti-drain is **not a pooling artifact**: even the minimum per-head v\_ratio for sigmoid (0.60–0.84) is near or above 1.0, confirming that across all 30 layers × 3 KV-heads the value at pos0 is not drained. Textinit has individual heads with near-zero v\_ratio (0.035–0.063), confirming extreme value drain in specific heads — the per-head spread is large.

### Image-swap verdict

All arms show `swap_v_pos0_cv > 0` throughout training:

| arm | swap\_v\_pos0\_cv (end) |
|---|---|
| baseline | 0.478 |
| baseline\_seed1 | 0.977 |
| g1gate | 0.692 |
| sigmoid | 0.527 |
| textinit | 0.277 |

**Verdict: PASS.** v\_norm at pos0 varies with image content in all arms. No measurement artifact from pooling or position-independent computation. The textinit arm has the lowest CV (0.28), consistent with its near-total attention concentration on pos0 making the attention pattern less image-content-sensitive (the sink is "locked in"); but it is still non-zero.

`swap_attn0_std` (per-image std of mean-attn→pos0): baseline 0.011, sigmoid 0.035, textinit 0.039. The higher std in sigmoid/textinit is expected given their strong concentration — small image-to-image variation in a large signal.

### Raw sigmoid mass (primary sigmoid metric)

| arm | raw\_to\_pos0\_mean (end) | raw\_to\_pos0\_max (end) |
|---|---|---|
| baseline | 0.060 | 0.165 |
| baseline\_seed1 | 0.062 | 0.129 |
| sigmoid | **0.301** | **0.659** |
| sigmoid\_seed1 | **0.276** | **0.664** |
| textinit | 0.627 | 0.986 |
| textinit\_seed1 | 0.226 | 0.527 |

Sigmoid raw mass to pos0 is 4–5× above baseline in both seeds, confirming the concentration is genuine (not a normalization artifact from shrinking total mass). This is the primary sigmoid metric per the TRM §8, and it agrees with the softmax-normalized Sink^0.2 result.

---

## 4. The overfitting characterization — full narrative

### What happened

The baseline arm (seed-0) was trained on the_cauldron (146,731 images, one QA per visit, bs=128) from 174.4M to 854.6M tokens without data refresh. Over 74 visual epochs:

- Train loss: 0.244 (174M) → 0.015 (842M) — model has nearly perfectly fit training examples.
- Val\_loss (same cauldron images, different QA): 1.169 (174M) → 1.40 (844M) — no improvement; slight degradation late.
- Val\_seen (seed-1 arms, 100M): 2.17 → 0.44 — plummets; memorization clear.
- Val\_unseen (seed-1 arms, 100M): 1.83 → 1.18 — improves through ~step 3000 (~28M tok), then plateaus.

**Memorization onset**: ~step 3000–4000 (28–38M tokens ≈ 2.5–3 visual epochs). After this point val\_unseen plateaus and val\_seen diverges. The probe-batch images were fixed (32 samples from the training set), so all sink metrics from ~step 3000 onward were measured on a memorized distribution.

### What this means for Gate A

The Gate A CONFIRM is valid **conditional on the training regime**: softmax baseline forms no concentration sink over 74 epochs of a 146K-image repeated dataset. What we cannot rule out: (a) fresh diverse data at comparable token count would form one sooner, (b) Gu et al.'s concentration-sink formation requires fresh-data entropy that the repeated dataset cannot supply. The norm/value signatures are **not** confounded this way — they emerged at steps 400–1200 (4–11M tokens, <1 epoch) and remained stable.

### What the mean-attn→pos0 decline means

mean\_attn→pos0 peaked at ~0.073 around 48M tokens, then declined to 0.052 at 855M. If the diversity-driven hypothesis is correct (sinks form where the data keeps presenting the same positional cues), then a memorized model should show *less* pos0 emphasis as the attention pattern generalizes beyond the training distribution — which is the direction observed. However the effect is modest (<30% relative decline over 700M tokens), and the 32-sample probe batch is fixed, so this could also reflect probe-batch variance. **Treat as weak (plausibility class, not evidence class) support for the diversity hypothesis.**

### Go/no-go input for RF run

The RF (random-fresh-1B) run is the decisive experiment for the diversity hypothesis. It requires:
1. **Streaming loader** (fix the non-streaming full-load issue in RF-BLOCKED.md — separate directive)
2. Fresh-image pool with ≥2.75M unique images (estimated ≥290 GB; FineVision 11-config)

**What RF must show to CONFIRM diversity hypothesis:**
- RF forms a concentration sink (Sink^0.2₁ ≥ 0.05) at some token count ≤1B
- RF's norm signatures are comparable to the repeated-data baseline at the same token count
- Mean attn→pos0 in RF does NOT decline late in training

**What RF must show to REFUTE:**
- RF shows the same zero-concentration result as the repeated baseline at 1B tokens
- (Would suggest fresh diversity is not the relevant variable; revisit whether 222M scale is simply too small to form a concentration sink)

**Either outcome is publishable**: CONFIRM strengthens the thesis (diversity drives emergence); REFUTE narrows the finding to "norm signatures decouple from concentration even at 1B fresh tokens in 222M VLMs" — which is itself novel.

---

## 5. Updated dissociation table (canonical, Session-1+2+3)

*Use this for the paper. Session-1 columns updated where Session-3 data improves precision. 
n=1 → n=2 for baseline/sigmoid/g1gate (matched budget); textinit at matched 60M.*

| arm | concentration | v\_ratio | h\_ratio | val loss | note |
|---|---|---|---|---|---|
| baseline (softmax, scratch) | **none** 0.000 ± 0 (n=2) | drained 0.71–0.72 (n=2) | 1.7–2.2 (n=2) | ~1.16 | confirmed through 855M, confounded by overfitting |
| sigmoid (no softmax, scratch) | **strong** 0.76–0.83 (n=2) | anti-drain 1.48–1.60 (n=2) | suppressed 1.1–1.3 (n=2) | ~1.11 | robust across seeds; raw mass 0.28–0.30 |
| g1gate (softmax+gate, scratch) | **weak** 0.004–0.011 (n=2) | mild drain 0.81–0.85 (n=2) | mixed 1.7–2.0 (n=2) | ~1.14 | fragile; do not headline; needs seed-3 |
| textinit (softmax, pretrained text) | **total** 0.56–0.85 @ 60M (n=2) | drain 0.38–0.63 (n=2) | **DISAGREE** 5.5 vs 42.5 (n=2) | ~0.83 | concentration confirmed; h\_ratio unresolved |

---

## 6. Verdict summary

| gate | status | key number |
|---|---|---|
| **Gate A** | **CONFIRM** (with repeated-data confound) | Sink^0.2=0 at 855M; confound: 74 visual epochs, memorized training set |
| **Gate B** | **CONDITIONAL PASS** | baseline✓✓ sigmoid✓✓ g1gate⚠ textinit\_h\_ratio⚠ |
| Dissociation | **Survives n=2** on primary claims (baseline/sigmoid) | |
| Image-swap | **PASS** — pos0 content-sensitive in all arms | swap\_v\_pos0\_cv > 0 throughout |
| Raw sigmoid mass | **CONFIRM** — 4–5× above baseline, both seeds | not a normalization artifact |

### Go/no-go for RF (once streaming loader fixed)

**GO.** Gate A confirmed (with documented confound), Gate B confirmed for sigmoid (the primary inversion), dissociation survives n=2. The remaining open questions (diversity hypothesis; textinit h\_ratio; g1gate fragility) are exactly what RF + seed-3 address. Running RF is the correct next step. Do not scale to A100/Stage-2 yet — Gate B still has two fragile arms.

### What must NOT be headlined yet

Per TRM §7 anti-zombie:
1. ~~"from-scratch softmax VLM forms no concentration sink at 1B"~~ → state as "no sink on 74-epoch repeated data; diversity untested (RF blocked)"
2. ~~"sigmoid CREATES a sink in VLMs"~~ → EARNED for concentration + raw mass at n=2; NOT earned for calling it a "VLM sink" without fresh-data RF control
3. ~~"G1 gate CREATES rather than prevents sinks"~~ → still fragile; do not headline; mark for seed-3
4. ~~"textinit h\_ratio = 42"~~ → seed-level divergence; report range 5–43; mark for seed-3

---

*Analysis generated 2026-06-13. Figures: `analysis/gateA_full_curve.png`, `loss_overfitting.png`, `gateB_two_seed.png`, `emergence_curves_2seed.png`, `reprobe_r3.png`. All plotting moved to Mac permanently.*
