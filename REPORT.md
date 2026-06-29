# When Do Attention Sinks Emerge in VLM Pretraining? — Session 1 (Stage-1 validation)

**Date:** 2026-06-12/13 · **Hardware:** 1× RTX 4090 48GB (vast.ai instance 40436103) ·
**Repo:** `nemesis8932/vlm-sink-emergence`, branch `sink-emergence` (fork of nanoVLM v0.1)

Executes Stage 1 of `plan_when-sinks-emerge-in-vlms.md`: build and validate the
sink-metric pipeline on from-scratch nanoVLM-222M training, with dense probing,
and collect first emergence curves for 4 arms.

## Setup

- **Model:** nanoVLM 222M = SigLIP-B/16-224 (85M, *pretrained init, trainable*) +
  SmolLM2-135M-architecture decoder (30 layers × 9 heads, GQA 3 kv-heads, d=576) +
  pixel-shuffle MP. Sequence = **49 image tokens (causal prefix) + 79 text tokens
  (left-padded) = 128**. Position 0 is the first *image* token — there is no BOS.
- **Data:** the_cauldron {vqav2, cocoqa, aokvqa, vsr} = 146,731 images; one QA pair
  sampled per image visit (≈740k unique QA texts). bs 128 (16,384 tok/step),
  bf16 autocast + torch.compile, ~54k tok/s.
- **Optimization:** AdamW wd=0.1 (Gu et al. setting), grad-clip 1.0, cosine LR with
  3% warmup: LM 4e-4 (from-scratch arms) / MP 2e-3 / ViT 1e-4.
- **Probe (every 100 steps):** fixed 32-sample batch; eager fp32 re-walk of the
  decoder, **validated against the model's own forward each call** (rel. err < 1e-2
  on valid positions). Metrics per (layer, head): mean attention→pos0 (row-normalized
  for the sigmoid arm), Sink^ε_1 for ε∈{0.2,0.3,0.4} (Gu et al.), attention mass by
  segment (pos0 / image / text), argmax key position, value-vector ℓ2 norm at pos0
  vs rest, residual-stream norm at pos0 vs rest. bf16 checkpoints at
  {0, 250, 1k, 2k, 4k, 8k, 12k, 16k, 24k}.

## Arms

| arm | attention | LM init | ViT init | wall budget |
|---|---|---|---|---|
| baseline | softmax | random | pretrained | 1.6 h |
| textinit | softmax | SmolLM2-135M pretrained | pretrained | 0.55 h |
| g1gate | softmax + G1 elementwise σ-gate (zero-init, post-SDPA) | random | pretrained | 1.0 h |
| sigmoid | unnormalized sigmoid (no softmax) | random | pretrained | 1.0 h |

## Results

*(figures in `analysis/`; full per-(layer,head) data in `runs/*/probes.jsonl`)*

### Headline: the VLM position-0 sink is inherited and amplified, not formed de novo

**From-scratch baseline (174.4M tokens, 18,287 steps):** the attention-concentration
sink **never fires** — Sink^0.2_1 = 0 throughout; the strongest single head reaches
only ~0.18 mean attention→pos0 (L1H1, plateauing from ~step 5k). Yet the *other* sink
signatures emerge early and persist: residual-norm ratio at pos0 crosses 2× by step
400 (~4M tokens) and stabilizes ≈2.2; value-norm ratio falls below 0.8 by step 1200
and settles ≈0.71–0.75. By step 10k, **80% of (layer,head) pairs have their argmax
attention key at position 0** (none in the text segment) — a broad, *flat* positional
preference without per-head mass concentration. Plan gate A (Sink^0.3 < 5% at this
budget) technically fires for concentration — but the norm/value machinery is
demonstrably in place, so "no sink" would be the wrong reading; the VLM-regime sink
at this scale is *soft*.

**Text-init (pretrained SmolLM2 decoder, 59.6M tokens):** the text LM's sink
relocates to the first **image** token *instantly* and alignment training amplifies
it explosively:

| step | Sink^0.2_1 | mean attn→pos0 | v_ratio | h_ratio |
|---|---|---|---|---|
| 0 | 0.193 | 0.13 | 0.73 | 3.5 |
| 100 | 0.552 | 0.25 | 0.48 | 10.5 |
| 500 | 0.830 | 0.39 | 0.34 | 20.7 |
| 1000 | 0.837 | 0.53 | 0.38 | 30.6 |
| 6244 (end) | 0.852 | **0.63** | 0.38 | 42.5 |

Mean attention to pos0 saturates at ~0.63 — *higher* than the ~0.47 Qiu et al. report
for text LLMs — on a token that is a SigLIP patch embedding, not BOS. The same
training data, architecture, and optimizer that produce **zero** concentration sink
from random init produce a near-total one from text-pretrained init.

### Decoupling (the plan's distinctive hypothesis) — supported in both directions

The headline dissociation is **massive-activation (h_ratio) rises while
attention-concentration (Sink) stays at zero**. Value-norm drain is *supporting*
context, not a co-lead: its drop is largely warmup and partially recovers, so it is not
framed as an emergence signal (see RF @1B below; consistent with `GATE_A_REPORT.md`).

- From-scratch: massive-activation emerges and grows (h_ratio>2 @ step 400) **without**
  attention concentration ever crossing ε=0.2 (174M tokens). Value-norm dips below 0.8
  (@ step 1200) but is non-monotone — it moves *opposite* to max_a0 in mid-training
  (steps 2k→4k: v_ratio 0.71→0.79 while max_a0 0.12→0.16), i.e. not a clean monotone drain.
- Text-init step 0: norm signatures (h_ratio 3.5, v_ratio 0.73) transfer to pos0
  ahead of concentration (Sink^0.2 = 0.19, Sink^0.3 = 0 at step 0; 0.83/0.75 by step 500).

**RF fresh baseline @ 1B (Gate A, 2026-06-15) — confirmed free of the repeated-data
confound.** On a fresh FineVision stream to 1B tokens, **Sink^0.3_1 = 0.000 across the
entire run** while **h_ratio rises 1.43→3.22 (+130%, init→1B, continuing post-warmup)**;
v_ratio ends at 0.69 but is non-monotone (~75% of its net drop is the 0–57M warmup, then
recovers). Fresh val_loss 1.46→0.638, no overfit. Massive-activation grows without any
concentration sink forming — the decoupling, on fresh data. Full rulings + figure:
`runs/rf_fresh_baseline/GATE_A_REPORT.md`, `runs/rf_fresh_baseline/decoupling_figure.svg`.

### Intervention arms

**G1 gate (Qiu et al.), from-scratch, 102.9M tokens / 10,786 steps.** The expectation
from the text-LLM literature — gating prevents sinks — inverts at this scale, in an
informative way:

- The gated arm is the **only from-scratch arm to develop ε-threshold sink heads**:
  first head >0.2 at step 3,400; up to 7 heads >0.2 (max attn→pos0 0.36 at step 6k;
  volatile 0.2–0.36 thereafter; 0.23 at end). The ungated baseline never crossed 0.2
  in 18k steps.
- Meanwhile its **norm signatures are consistently milder** than baseline at matched
  steps: h_ratio ≈ 1.6–1.9 (baseline 2.1–3.2), v_ratio ≈ 0.81–0.85 (baseline
  0.70–0.79).
- Val loss is consistently ~0.03–0.1 better than baseline at matched steps
  (1.138 vs 1.161 at end, and the gap is larger mid-training), echoing Qiu's quality
  result.

Interpretation: with an output gate available, the model can *afford* attention
concentration — the gate suppresses whatever the sink position would inject into the
residual stream, so concentration no longer has to co-occur with value-norm drain or
massive activations. The gate doesn't prevent the sink; it **decouples the
concentration signature from the norm signatures** (the same decoupling the
from-scratch baseline shows in the opposite configuration). The Qiu et al. "4.8%
first-token attention" result was measured on text-pretrained models at far larger
scale; in the from-scratch VLM regime the gate's effect on *where attention goes* is
the opposite, at least within 100M tokens.

**Sigmoid attention (Gu et al., no softmax), from-scratch, 101.7M tokens / 10,664
steps.** Gu et al. report that unnormalized sigmoid attention *prevents* both sinks
and massive activations up to 1B params in text LMs. In the from-scratch VLM regime
the result is the most surprising of the study:

- Relative concentration on pos0 emerges **fastest of any from-scratch arm**:
  Sink^0.2_1 crosses >0 at step **500** (g1gate 3,400; baseline never), reaching
  0.83 of heads >0.2 and 0.66 >0.3 by the end; max head attn→pos0 0.87. We verified
  this is *not* a normalization artifact — the raw (unnormalized) mean sigmoid score
  to pos0 is 0.25 vs ~0.05 uniform, i.e. pos0 genuinely receives elevated gate-open
  mass, not just a large share of a shrinking total.
- But the **norm signatures invert**: value-norm at pos0 is *amplified* to ≈1.5×
  the rest (anti-drain; baseline drains to 0.71), and there are **no massive
  activations** (h_ratio ≈ 1.0–1.15, vs baseline 2.2, textinit median ~12× / range 5.5–42.5). This matches the
  half of Gu's claim that survives — sigmoid kills the residual-norm blowup — while
  *contradicting* the "no sink" half at the level of attention concentration.
- Best val loss of the from-scratch arms (1.108).

So sigmoid attention does not remove the sink; it produces a **different sink
object** — attention concentration on pos0 *without* the value/norm pathology that
the literature treats as part of the same phenomenon.

## Synthesis: the four-way dissociation

The central question of the plan — do attention concentration and the norm signatures
(led by massive-activation) **decouple** during VLM pretraining? — gets a clear yes. Holding data, optimizer, and
the 222M architecture fixed, each lever produces a *different combination* of the
sink signatures, proving they are independently controllable rather than facets of
one phenomenon:

| arm | concentration (Sink^0.2_1) | value-norm @ pos0 | massive activation (h_ratio) | val |
|---|---|---|---|---|
| baseline (softmax, scratch) | **none** (0.00) | drained (0.71) | moderate (2.2) | 1.161 |
| g1gate (softmax+gate, scratch) | weak (0.004, volatile) | mild drain (0.82) | **suppressed** (1.6) | 1.138 |
| sigmoid (no softmax, scratch) | **strong** (0.83) | **amplified** (1.48) | **none** (1.1) | 1.108 |
| textinit (softmax, pretrained text) | **total** (median 0.58, range 0.56–0.85) | **severe drain** (0.38–0.63) | **extreme** (median ~12×, range 5.5–42.5) | 0.832 |

Lead–lag (first probe step a signature crosses threshold): in the from-scratch
baseline the **norm signatures lead and the concentration sink never arrives**
(h_ratio>2 @ step 400, v_ratio<0.8 @ 1200, concentration never crosses 0.2 in 174M
tokens). With text init, **all signatures are present at step 0** — the sink is
inherited from text pretraining, relocated onto the first *image* token, and
amplified by alignment, never formed de novo. This is direct evidence for the
plan's disentanglement of *inherited vs formed-fresh* sinks.

## Compute used this session

RTX 4090 @ $0.7037/hr. Productive training: baseline 96 min + textinit 33 + g1gate 60
+ sigmoid 60 = **249 min ≈ 4.15 GPU-h ≈ $2.92**. Setup/download/benchmark ≈ 37 min
(~$0.44). Avoidable waste was small: ~3 OOM probes at bs256/512 and one baseline
relaunch (bug catch) ≈ 12 min (~$0.14). The largest waste was the **idle tail** after
all arms finished at 02:30 UTC: the 03:40 backstop fired but `vastai stop` did not
take effect, leaving the instance idle ~2.2 h (~$1.5) until manually stopped.

### Step-0 sanity observations (already informative)

- **baseline/g1gate (random init):** mean attn→pos0 ≈ 0.053 ≈ uniform-causal
  expectation; Sink^0.2 = 0; v_ratio ≈ 1.0. No sink at init, as expected.
- **textinit (pretrained SmolLM2 decoder, image prefix):** at step 0 the *text* LM's
  sink machinery partially transfers to the multimodal sequence: residual-stream norm
  at pos0 is **3.2×** the rest (massive-activation transfer to an *image* token), and
  value-norm ratio at pos0 is **0.64** (the value-drain signature), yet max attn→pos0
  is only 0.17 (Sink^0.2_1 = 0) — i.e. **the norm/value signatures transfer to
  position 0 before the attention-concentration signature does**. Within a handful of
  optimization steps attn→pos0 begins rising. This is a step-0 decoupling data point.

## Caveats / Limitations

- **Per-position anchoring.** All three signatures are measured at the first image token
  (pos0). Per-position attention mass is confirmed across all arms and seeds (seed-0: HF
  npz reprobe; seeds 1/2 + RF: local streaming reprobe, commit 158cd5f). **pos0 is the
  max-mass token for baseline, g1gate, sigmoid (all seeds), and textinit seed-0/seed-1**;
  the reported seed-0 magnitudes (incl. textinit h_ratio 42.5) are correctly anchored.
  Two residual flags folded into v1 limitations:
  *(a) textinit seed-2 — the three signatures sit on DIFFERENT tokens (spatial decoupling,
  not an anchoring artifact).* The per-position **norm** dump (commit `d7b5fc4`,
  `analysis/per_position_norms.json`) shows: attention max-mass at pos1, but the
  massive-activation (‖h‖) peak at pos13 and the value-drain trough also at pos13 — none
  co-located. Anchoring h_ratio under the *attention* sink (pos1) gives 9.46, **lower**
  than the pos0 value (14.7), so pos0-anchoring is **not** an under-measurement; the earlier
  "lower-bound / anchor-mislocation" hedge is **withdrawn**. The h_ratio spread across seeds
  (44.8 / 8.2 / 14.7 for s0/s1/s2) is therefore genuine seed variance, not a measurement
  position artifact. (Note: the ‖h‖ peak is *not* fixed at pos0 either — s1 peaks at pos1,
  s2 at pos13 — so massive-activation does not track the attention sink; it is the cleaner,
  stronger reading that the signatures decouple in *position* as well as magnitude.)
  *(b) RF* — argmax-vote (76% of heads at pos1) is diffuse noise: pos1 = 10.0% of mass
  vs pos0 = 8.3%, both baseline-class (compare sigmoid pos0 = 30%+); no off-pos0
  concentration, Gate-A absent verdict robust.
  Figures: `analysis/per_position_mass_seed0.svg`, `analysis/argmax_position_by_arm.svg`;
  per-position norm data `analysis/per_position_norms.json`.
- **Pretrained ViT** = partial-from-scratch + inherited-norm confound; defense: h_ratio starts
  ≈1.0–1.4 and *rises* (formed, not inherited). Random-ViT variant is future work.
- **Token scale** ≤1B/arm vs Gu-scale ~5B (emergence is early; larger-scale is future work).
- **textinit magnitude** is seed-sensitive — same seed ordering (s0 ≫ s2 > s1) on two probe
  batches: live-probe (tail-1024) h_ratio range 5.5–42.5 (table above); reprobe
  per-position-norm batch (streaming-32) 44.8 / 8.2 / 14.7. seed-0 is the outlier on every
  signature; reported as range/median, the corner is a *kind*, not a calibrated magnitude.
  Confirmed genuine seed variance, not an anchoring artifact (see per-position bullet above).
- **Provenance:** seed-0 raw not re-derivable first-hand (trusted from checksummed
  `archive/session3/`); seed-2 run only for g1gate + textinit.
- Domain shift (the_cauldron→FineVision; ADR-0001) accepted to remove the repeated-data
  confound; domain-matched fresh-repeated control is the known follow-up.
- MMStar accuracy not measured (GPU budget went to emergence curves).
