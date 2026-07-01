# Sink Signatures Dissociate During From-Scratch Vision-Language Pretraining

*Draft v1 — assembled 2026-07-02. Not for submission until confirmed by the author (see note at end).*

# Abstract

Attention concentration on early tokens, near-zero value-vector norms at those tokens, and
massive residual-stream activations are widely observed to co-occur in language models and
are often treated, implicitly, as facets of a single "attention-sink" phenomenon. We track
these three signatures separately during from-scratch pretraining of a 222M-parameter
vision-language model (a SigLIP-B/16 encoder feeding a randomly initialized
SmolLM2-135M-architecture decoder), varying four training levers — standard softmax
attention, an output-gated softmax variant, an unnormalized sigmoid/no-softmax variant, and
decoder initialization from a pretrained text LM. Across three seeds, the four levers land in
four distinct regions of (concentration × value-norm × massive-activation) space: no two arms
share the same signature triple, and value-norm ratio alone takes three qualitatively
different directions (drained, unchanged, amplified) depending on the lever. On a
confound-free, 1-billion-token fresh-data run, massive activation grows 130% while attention
concentration stays at exactly zero for the entire run. Per-head correlation between
concentration and value-norm flips sign across arms (+0.67 to −0.76; pooled −0.20),
inconsistent with a single mechanism driving both. We are not the first to decouple sink
signatures — prior text-only work already separates massive activations from concentration —
but we show, for the first time in a from-scratch multimodal pretraining setting, that all
three signatures, including value-norm drain as a third independent axis, are separately
controllable. We report these results together with their limitations at 222M-parameter,
1B-token scale.

# 1. Introduction

Decoder-only language models reliably develop a peculiar behavior early in training: a
disproportionate share of attention mass concentrates on the first token(s) of the sequence,
regardless of content. In text language models, this **attention concentration** has been
observed to co-occur with two other symptoms — near-zero value-vector norms at the same
token ("value-state drain") and abnormally large residual-stream activations ("massive
activations") — concentrating on the same few tokens (Gu et al., arXiv:2410.10781) and
co-emerging early in training, by roughly step 1k (arXiv:2510.06477). Their consistent
co-occurrence has led these signatures to be treated, often implicitly, as facets of a
single "attention-sink" phenomenon. One recent line of work goes further and argues the
signatures are *causally* unified through massive activations in the residual stream
(Queipo-de-Llano et al., arXiv:2510.06477); another shows, in text models, that at least two
of the three symptoms are in fact *dissociable* via architectural interventions (Sun et al.,
arXiv:2603.05498; Chen & Yao, arXiv:2603.17771) — evidence the field is still working out
which parts of this picture are one mechanism and which are several.

Almost all of this evidence comes from text-only language models, analyzed either at a single
trained checkpoint or across pretraining of a decoder-only architecture with no visual input.
Vision-language models (VLMs) add a second modality with its own sink-like behavior — vision
transformers grow their own high-norm "register" tokens (Darcet et al.) — and recent LVLM
studies show vision-side and language-side sinks have distinct origins (Luo et al.,
arXiv:2510.08510; Choi et al., arXiv:2604.03316). But those studies analyze frozen, already
-trained backbones. No prior work, to our knowledge, tracks attention concentration,
value-norm drain, and massive activations **simultaneously, as three separately-measured
quantities, across from-scratch multimodal pretraining** — which is the only setting where
one can ask whether the signatures emerge together, in sequence, or independently, rather than
merely co-occur in a model that already has them.

We train a 222M-parameter nanoVLM (SigLIP-B/16 vision encoder + a SmolLM2-135M-architecture
decoder) from random initialization under four training levers chosen specifically to target
one sink-relevant mechanism each: standard softmax attention, an output-gated softmax variant
(Qiu et al., arXiv:2505.06708), an unnormalized sigmoid attention variant with no softmax
normalization (Gu et al.), and initializing the decoder from a pretrained text LM rather than
random weights. We track concentration, value-norm ratio, and massive activation as three
separate quantities throughout pretraining, and additionally re-test the no-concentration
result on a confound-free, 1-billion-token fresh-data run to rule out a repeated-data
overfitting artifact.

Our contributions:

1. **Four-way dissociation across three seeds.** The four training levers land in four
   distinct corners of (concentration × value-norm × massive-activation) space; no two arms
   share a signature triple, and value-norm ratio alone takes three qualitatively different
   directions (drained, unchanged, amplified) depending on the lever.
2. **Confound-free decoupling at 1B tokens.** On a fresh, non-repeated data stream, massive
   activation rises 130% over a full billion tokens of training while attention concentration
   stays at exactly zero for the entire run.
3. **No universal head-level coupling.** The per-head correlation between concentration and
   value-norm flips sign across arms (+0.67 to −0.76; pooled −0.20), inconsistent with a
   single shared mechanism driving both.

We scope our novelty claim carefully: we are **not** the first to decouple sink signatures —
prior text-only work already separates massive activations from concentration via
normalization and value-path interventions (Sun et al.; Chen & Yao). Our contribution is the
**conjunction**: showing all three signatures, including value-norm drain as a third
independently-moving axis, are separately controllable during **from-scratch multimodal**
pretraining, a setting no prior decoupling or co-emergence study has examined.

# 3. Method

**Model.** All runs use a 222M-parameter nanoVLM: a SigLIP-B/16 vision encoder (pretrained,
frozen initialization but trainable) feeding a decoder with the SmolLM2-135M architecture.
Except where noted, the decoder is trained **from random initialization** — the point of the
study is to observe sink signatures form during pretraining, not to probe an already-trained
model. Sequences are 49 image tokens (causal prefix) followed by 79 left-padded text tokens
(128 total). Critically, **position 0 is the first image token, not a BOS token** — there is
no text BOS in this layout, so any sink behavior at pos0 is a property of the visual prefix,
not inherited BOS machinery.

**Arms (training levers).** Four arms isolate which lever produces which sink signature:

| arm | attention | LM init | ViT init |
|---|---|---|---|
| baseline | softmax | random | pretrained |
| g1gate | softmax + elementwise sigmoid gate (zero-init, applied post-SDPA; Qiu et al. G1) | random | pretrained |
| sigmoid | unnormalized sigmoid, no softmax (Gu et al.) | random | pretrained |
| textinit | softmax | pretrained SmolLM2-135M | pretrained |

A fifth configuration, **RF** (random-fresh baseline), is the `baseline` arm retrained on a
fresh (non-repeated) FineVision image stream at 1B tokens, to re-test the no-concentration
result free of a repeated-data overfitting confound present in an earlier run (below).

**Data.** The four-arm comparison trains on `the_cauldron` (four curated subsets, ~146K
images), matched to ~100M tokens/arm (60M for textinit, which reaches its 3-way signature
floor earlier — see Results). This regime is known to reach high visual-epoch counts by
1B tokens; RF instead streams FineVision (~4.6M natural images, ~1–2 visual epochs at 1B
tokens) to remove that repetition confound. Swapping datasets introduces a domain-shift
confound in its place, which we accept and document rather than run a third,
domain-matched control arm (`docs/adr/0001`).

**Metrics — three signatures, tracked separately.** The three sink symptoms often reported
together in the text-LM literature are logged as independent quantities per (layer, head):

- **Concentration**: `Sink^ε_1` = fraction of (layer, head) pairs with mean attention→pos0
  > ε (ε=0.3 default; robustness checked at 0.2/0.4 — both thresholds appear in the results
  below depending on source table), plus max attention→pos0.
- **Value-norm ratio** (`v_ratio`): ‖v‖ at pos0 vs. the mean over the rest of the sequence.
- **Massive activation** (`h_ratio`): residual-stream norm at pos0 vs. rest.

All three are **pos0-anchored by construction**: this is a deliberate, stated measurement
choice (position 0 is the argmax-attention token in every arm at seed-0 — confirmed
first-hand, see Results §4.3 / Limitations), not an assumption baked in silently.

**Probe.** `probe_sinks()` re-walks the decoder in eager/fp32/no-grad mode from the live
module weights, independent of the training path (SDPA kernels, autocast, `torch.compile`);
every call is validated against the model's real forward pass (relative error < 1e-2), so the
probe cannot silently drift from what the model is actually doing. The probe batch is fixed
(seed 0), making metrics comparable seed-to-seed and run-to-run. Probes run every 100
training steps.

**Training recipe.** AdamW, weight decay 0.1 (following Gu et al.), gradient clip 1.0,
cosine learning-rate schedule with 3% warmup (LM 4e-4 / modality-projector 2e-3 / ViT 1e-4),
bf16 autocast, `torch.compile`. All arms in a comparison are trained with byte-identical
recipes and token budgets except for the lever under test.

# 4. Results

## 4.1 Four training levers land in four distinct signature corners

Table 1 reports the three signatures for all four arms across n=2–3 seeds (baseline/sigmoid
n=2; g1gate/textinit n=3), matched to ~100M tokens (textinit stopped at its 60M three-way
floor; seed-0 numbers trusted from a checksummed archive, not re-derivable first-hand — see
Limitations).

**Table 1 — signature triple by arm, all seeds.**

| arm | seed | tokens | Sink^0.2_1 | mean attn→pos0 | v_ratio | h_ratio |
|---|---|---|---|---|---|---|
| baseline | s0 | 101M | 0.000 | 0.068 | 0.723 | 2.16 |
| baseline | s1 | 100M | 0.000 | 0.062 | 0.687 | 1.71 |
| g1gate | s0 | 101M | 0.004 | 0.068 | 0.805 | 1.67 |
| g1gate | s1 | 100M | 0.011 | 0.073 | 0.845 | 2.04 |
| g1gate | s2 | 100M | 0.0037 | 0.072 | 0.854 | 2.24 |
| sigmoid | s0 | 102M | 0.830 | 0.377 | 1.479 | 1.10 |
| sigmoid | s1 | 100M | 0.756 | 0.311 | 1.603 | 1.30 |
| textinit | s0 | 60M | 0.852 | 0.627 | 0.377 | 42.5 |
| textinit | s1 | 60M | 0.556 | 0.235 | 0.634 | 5.50 |
| textinit | s2 | 60M | 0.578 | 0.232 | 0.484 | 12.2 |

Collapsing to per-arm ranges (Table 2), **no two arms share a signature triple**, and value-norm
ratio alone spans three qualitatively different behaviors — drained below 1, essentially
unchanged near 1, and amplified above 1:

**Table 2 — the four corners.**

| arm | concentration | value-norm | massive activation |
|---|---|---|---|
| baseline | absent (0.000) | mild drain (<1, 0.69–0.72) | moderate (h 1.7–2.2) |
| g1gate | suppressed (0.004–0.011) | **none** (~1, 0.81–0.85) | moderate (h 1.7–2.2) |
| sigmoid | strong (0.76–0.83) | **amplified** (>1, 1.48–1.60) | none (h 1.1–1.3) |
| textinit | total (0.56–0.85) | strong drain (0.38–0.63) | extreme (h 5.5–42.5) |

g1gate separates from baseline purely on value-norm (concentration suppressed either way,
massive activation unchanged); sigmoid and textinit separate from each other on both the
*direction* of the value-norm effect (amplified vs. drained) and massive-activation magnitude.
This four-way spread is what we mean by "signatures dissociate": a single training lever can
move one axis while leaving the others essentially where a different lever left them.

Reproducibility differs by arm. g1gate is the tightest: Sink^0.2_1 = 0.004 / 0.011 / 0.0037
across three seeds, all far below the sink threshold, with mean/max attention→pos0 flat
across seeds. textinit's *corner* — high concentration, strong value-drain, large h_ratio —
is robust across all three seeds, but its **magnitude is not**: seed-0 is the high outlier on
every signature simultaneously (Sink 0.85 vs. 0.56–0.58; mean attn→pos0 0.63 vs. 0.23; h_ratio
42.5 vs. 5.5–12.2), while seed-1 and seed-2 track each other closely. h_ratio has plateaued by
60–100M tokens in both low-outlier seeds, so the spread reflects genuine seed sensitivity
rather than an unconverged transient. We therefore report textinit's massive-activation as a
**range (5.5–42.5×) with median ~12×**, not a point estimate, throughout this paper.

## 4.2 Massive activation grows for 1B fresh tokens with concentration pinned at zero

The four-arm comparison above uses `the_cauldron`, a fixed 146K-image pool that reaches high
visual-epoch counts by 1B tokens — a possible confound for any "no concentration sink"
reading. We re-ran the `baseline` arm (softmax, random decoder, pretrained ViT, seed 0) on a
**fresh**, non-repeated FineVision stream to 1B tokens (RF; "Gate A" retest).

**Sink^0.3_1 = 0.000 across the entire 0→1B run** — the concentration sink does not emerge at
any point in this trajectory, confirming the earlier repeated-data result was not an artifact
of overfitting. Meanwhile massive activation keeps rising well past warmup:

| signature | @ init | @ 1B tokens | net change |
|---|---|---|---|
| Sink^0.3_1 (concentration) | 0.000 | 0.000 | flat zero, entire run |
| max attn→pos0 | 0.056 | 0.098 | ~flat, far below sink threshold |
| h_ratio (massive activation) | 1.43 | 3.22 | **+130%**, rise continues post-warmup |
| v_ratio (value-norm) | 1.00 | 0.69 | net decrease, non-monotone |

h_ratio's rise is not a warmup artifact: it continues monotonically from 2.40 at ~57M tokens
to 3.22 at 1B, well past the 3% warmup window. v_ratio ends below its init value but ~75% of
that drop occurs during the 0–57M warmup segment and partially recovers afterward, so we do
not frame it as ongoing emergence — it is reported as supporting context, not a second
headline. The clean dissociation here is: **massive activation continues to grow for a full
billion tokens of training while attention concentration never leaves zero.**

## 4.3 No universal head-level coupling between concentration and value-norm

If concentration and value-norm drain were expressions of one underlying mechanism, their
per-head correlation should have a consistent sign across training regimes. Instead, Pearson
r between per-head attention→pos0 and value-norm ratio, computed at the final checkpoint of
each arm (n=270 heads/arm), **flips sign by arm**:

| arm | baseline | g1gate | sigmoid | textinit | RF | pooled (n=1350) |
|---|---|---|---|---|---|---|
| r | +0.67*** | +0.53*** | −0.03 ns | −0.76*** | +0.43*** | −0.20*** |

This is not a null/uncorrelated cloud — most individual arms show strong correlations — but
the *sign* is arm-dependent (+0.67 in baseline vs. −0.76 in textinit), and the pooled
correlation is weak only because the arms cancel each other. A single shared mechanism
predicts a consistent sign; what we observe is inconsistent with that, and consistent with
each lever inducing its own local relationship between the two axes.

## 4.4 Ordering and location of the signatures

Lead-lag analysis across training steps shows a consistent ordering by arm family: in the
softmax-from-scratch arms without gating (baseline), the norm-based signatures (value-norm,
massive activation) cross their thresholds early while concentration never arrives at all;
sigmoid mirrors this — concentration crosses early while the norm signatures stay flat; and
textinit inherits all three signatures already present at initialization, since it starts
from a pretrained text LM. Query-by-key attention maps of each arm's top concentration head
show the pos0 stripe **absent in baseline, total in textinit, and already present at
initialization in textinit** (inherited from the text LM checkpoint) — we make no claim about
the sigmoid arm's attention-map stripe from this figure specifically, since the checkpoint
dump for sigmoid selected its top head by an unnormalized gate score rather than the true
maximum-concentration head; sigmoid's true concentration is established instead by the
Sink^ε_1 numbers in §4.1 and the entropy-collapse result below. Attention-entropy collapse
(a text-LM correlate of concentration) tracks concentration, not the norm signatures: only
sigmoid and textinit show entropy collapse, while baseline, g1gate, and RF stay flat.

As a supporting observation (not a second headline result): in textinit, the three
signatures are not only decoupled in magnitude but partially decoupled in *position* — across
seeds, the massive-activation peak and the value-drain minimum are not always co-located with
the attention sink itself (e.g., attention concentrating at pos1 while the residual-norm peak
and value-drain sit at pos13 in one seed). We report this as a caveat on where exactly to
place the pos0-anchored magnitudes for textinit (see Limitations), not as an independent
finding.

# 5. Related Work

**Attention sinks in text LMs.** Gu et al. (arXiv:2410.10781) characterize the attention sink
as acting "more like key biases, storing extra attention scores, which could be
non-informative and not contribute to the value computation" — i.e. they report small
key/value norms at the sink token as part of the same phenomenon, and connect it to massive
residual-stream activations (Sun et al., 2024; Cancedda, 2024). They also show that
replacing softmax attention with an unnormalized sigmoid variant prevents sink formation up
to 1B parameters at near-zero cost to validation loss — the result our `sigmoid` arm is built
on. Guo et al. (2024) describe the same coupling as "active-dormant" attention heads, where a
sink head's value state is actively driven toward zero. Queipo-de-Llano et al.
(arXiv:2510.06477) go further and argue for genuine **causal unity**: they show massive
activations mathematically require representational compression, and that ablating a model's
layer-0 massive activation eliminates both compression and sink formation; across Pythia
checkpoints they find all three of their phenomena (sinks, compression valleys, massive
activations) emerge together by step ~1k and stay synchronized thereafter. We reserve
"causally unified" strictly for this result — it is stronger and more specific than the
co-occurrence claims above.

**Prior decoupling work (text-only, two-way).** Two recent papers already separate pairs of
these signatures in text models, and our claim must be scoped around them. Sun, Canziani,
LeCun & Zhu (arXiv:2603.05498) show massive activations and attention sinks are *dissociable*
architectural artifacts: varying normalization scheme (Sandwich, DynamicTanh, QKNorm vs.
Pre-Norm) crushes the massive-activation spike while the sink ratio survives, arguing the two
signatures' frequent co-occurrence is "largely an architectural artifact ... rather than a
deep functional coupling." This is a **two-way** decoupling (massive activation vs.
concentration), text-only, via a normalization lever — it does not include value-norm drain
as an independent axis, and it analyzes trained checkpoints rather than from-scratch emergence
dynamics. Chen & Yao (arXiv:2603.17771) decouple the same pair from the opposite direction,
from scratch: in 0.1B/0.3B text LMs probed at dense checkpoints, a value-scale intervention
preserves attention sinks while suppressing massive activations. Neither paper treats
value-norm drain as a third, independently-moving axis, and neither is multimodal.

**Multimodal sink work (inference-time, frozen backbones).** Two LVLM papers study sinks in
already-trained, frozen models rather than from-scratch pretraining dynamics. Luo et al.
(arXiv:2510.08510) identify high-norm attention-sink tokens in the ViT encoder and separate
ViT-propagated sinks from LLM-emerged ones at inference time. Choi et al. (arXiv:2604.03316)
similarly distinguish vision-sinks from language-sinks in a frozen LVLM backbone and propose
a layer-wise sink-gating intervention. Both establish that multimodal sinks have distinct
vision- and language-side origins — a useful frame for our result — but neither tracks
signature emergence across from-scratch pretraining, and neither performs a three-way
decoupling.

**The G1 gating lever.** Qiu et al. (arXiv:2505.06708) introduce a head-specific, zero-init
elementwise sigmoid gate applied to the attention output (post-SDPA, before the output
projection); in text LLMs this "largely reduces the attention score allocated to the first
token and decreases massive activations," improving downstream quality. Our `g1gate` arm
applies this lever in the multimodal from-scratch setting; a companion paper from the same
group (Qiu et al., arXiv:2601.22966) argues, in the opposite direction, that outlier-driven
rescaling from attention and residual sinks is *essential* to stable training — evidence the
field is actively unsettled on whether these signatures should be suppressed or preserved.

**Positioning.** Taken together, the closest prior work separates at most two of our three
axes, in text-only models, via post-hoc normalization or value-path interventions on trained
checkpoints. We are not aware of any study that tracks all three signatures — concentration,
value-norm, and massive activation — as independently controllable quantities during
from-scratch **multimodal** pretraining. We frame our contribution accordingly: **first to
show all three sink signatures independently controllable during from-scratch multimodal
pretraining, adding value-norm drain as a third axis** beyond the massive-activation-vs-sink
dissociation already shown in text models (Sun et al.; Chen & Yao). We do not claim to be
first to decouple sink signatures in any setting.

# 6. Limitations

**(1) Pos0-anchored metrics.** All three signatures are measured relative to the first
image token (position 0). We verified this anchoring first-hand at seed-0 by checking
per-position attention mass directly (not just argmax): pos0 is the maximum-mass token in
every arm — baseline (mean mass 0.06, max-head 0.17), g1gate (0.07, 0.23), sigmoid (0.30,
0.66), and textinit (0.63, 0.99, with the next-highest position, pos1, at 0.009) — which
validates the pos0-anchored magnitudes reported here, including textinit's seed-0 h_ratio of
42.5. This closes the concentration-*location* question for seed-0. It does not close it for
seed-1/seed-2: live-probe argmax data at those seeds shows the most-attended position can
migrate away from pos0 (e.g., one textinit seed's argmax mass splits across pos1 and pos13),
which is consistent with — but not yet directly confirmed by — a per-position norm scan at
those seeds. We therefore report textinit's pos0-anchored magnitudes as likely a
representative but not perfectly calibrated read of that arm's true massive-activation and
concentration magnitude, and report them as a range/median (§4.1) rather than a point
estimate. A full per-position value/residual-norm scan across all seeds is left to a
camera-ready revision.

**(2) Pretrained vision encoder.** The SigLIP encoder is pretrained, so no arm is fully "from
scratch" — only the decoder is. Vision transformers are known to grow their own high-norm
register/artifact tokens independent of any decoder behavior (Darcet et al.; arXiv:2510.08510
separates ViT-propagated sinks from LLM-emerged ones), so part of our observed
massive-activation signal could in principle be inherited from the vision encoder rather than
formed by the decoder during training. Our defense: h_ratio starts near 1.0–1.4 at
initialization and *rises* substantially during training (RF: 1.43→3.22 over 1B tokens);
pure inheritance from a static pretrained encoder would instead predict a high, roughly
constant h_ratio from step 0. A `--vit_init random` control, isolating the decoder's
contribution entirely, is future work.

**(3) Token scale.** Our runs reach at most 1B tokens per arm, versus the ~5B canonical
budget used in the text-LM sink literature (Gu et al.). Sink emergence in text models is
early relative to that budget, and 1B tokens is past the point where text-LM sinks typically
form, but we cannot rule out that a signature we call "absent" at 1B would eventually emerge
at larger scale; larger-scale confirmation is future work.

**(4) Text-init magnitude reproducibility.** The `textinit` arm's massive-activation
magnitude is seed-sensitive: h_ratio ranges 5.5–42.5 across three seeds, with seed-0 a
consistent outlier on every signature (concentration, value-drain, and massive activation
simultaneously). We report textinit signatures throughout as a range with median (~12× for
h_ratio), not a single point, and treat the *corner* (high concentration + strong drain +
large massive activation) rather than any specific magnitude as the reproducible claim.

**(5) Provenance and seed count.** Seed-0 raw probe data for the four-arm comparison is not
independently re-derivable on the auditing machine — it is trusted from a checksummed archive
summary rather than re-computed first-hand from raw logs, though seed-1/seed-2 were
independently re-derived and confirmed the seed-0-derived headline numbers are not artifacts
of a metric-labeling error we separately caught and corrected (§4.1). baseline and sigmoid
have only two seeds each; g1gate and textinit have three. The confound-free 1B-token fresh
-data run (§4.2) is a **single seed**; concentration was already reproducibly zero across two
seeds on the repeated-data version of this arm, which we take as sufficient support for the
negative (no-concentration) claim, but a second fresh-data seed would strengthen it.

**(6) Domain shift in the fresh-data control.** The fresh-data run removes a repeated-data
overfitting confound (the original comparison data reaches ~74 visual epochs by 1B tokens) by
switching from `the_cauldron` to FineVision, which introduces a domain-shift confound in its
place (`docs/adr/0001`). We accept this trade because the repetition confound is dominant and
the fresh data pool is chosen to minimize domain shift (natural-image, COCO-heavy selection,
<3% subset overlap with the repeated pool); a domain-matched fresh-*and*-repeated control arm
is the known follow-up if this is contested.

**No benchmark accuracy is reported.** We measure sink signatures via the probe described in
§3; we did not run downstream benchmark evaluation (e.g., MMStar) on any arm in this study,
and make no claim about how signature dissociation relates to downstream task accuracy.

# 7. Conclusion

We tracked three signatures commonly treated as facets of one "attention-sink" phenomenon —
attention concentration, value-norm drain, and massive activation — separately, across
from-scratch multimodal pretraining. Four training levers produced four distinct signature
corners, no two arms sharing a triple; on a confound-free 1-billion-token fresh-data run,
massive activation kept rising while concentration stayed exactly zero for the entire run;
and per-head correlation between concentration and value-norm flipped sign across arms. None
of this establishes that the three signatures are unrelated in general — text-LM work shows
real interactions between them, and one recent study argues for a shared causal origin in
massive activations (Queipo-de-Llano et al.) — but it does show that in from-scratch
multimodal pretraining, the three axes can be moved independently by simple training-time
levers, extending prior two-way text-only decoupling results to a third axis and a new
setting. Natural next steps: a random-initialized vision encoder to fully isolate the
decoder's contribution to massive activation, a per-position value/residual-norm scan across
all seeds to close the remaining anchoring caveat on the text-init arm, and scaling the fresh
-data run past 1B tokens to match the token budgets used in the text-LM literature.

# Appendix

## A. Supporting figures

- **Figure A1** (`analysis/fig1_layerhead_grid.svg`) — per-(layer, head) attn→pos0 over
  training, row-normalized: baseline/RF stay cold throughout, textinit is hot from
  initialization, sigmoid lights up a mid-network band. Orienting figure for where in the
  network concentration lives.
- **Figure A2** (`analysis/fig3_perhead_scatter.svg`) — per-(layer, head) concentration vs.
  value-norm ratio, colored by arm; the correlation sign flips by arm (Table 3 below). Do
  **not** read this as an uncorrelated cloud — most individual arms show strong |r|; only the
  pooled correlation is weak, because the arms' signs cancel.
- **Figure A3** (`analysis/fig5_entropy.svg`) — attention-entropy collapse over training,
  by arm: only sigmoid and textinit collapse; baseline, g1gate, and RF stay flat. Ties the
  concentration axis to the entropy-collapse framing used in the text-LM literature.

## B. Full per-seed signature table

(Reproduced from Table 1, §4.1, with max attention→pos0 added where available — this metric
was only logged for seed-1/seed-2, not seed-0, so it is kept out of the main comparative
table.)

| arm | seed | tokens | Sink^0.2_1 | mean attn→pos0 | max attn→pos0 | v_ratio | h_ratio |
|---|---|---|---|---|---|---|---|
| baseline | s0 | 101M | 0.000 | 0.068 | — | 0.723 | 2.16 |
| baseline | s1 | 100M | 0.000 | 0.062 | — | 0.687 | 1.71 |
| g1gate | s0 | 101M | 0.004 | 0.068 | — | 0.805 | 1.67 |
| g1gate | s1 | 100M | 0.011 | 0.073 | ~0.21 | 0.845 | 2.04 |
| g1gate | s2 | 100M | 0.0037 | 0.072 | ~0.22 | 0.854 | 2.24 |
| sigmoid | s0 | 102M | 0.830 | 0.377 | — | 1.479 | 1.10 |
| sigmoid | s1 | 100M | 0.756 | 0.311 | — | 1.603 | 1.30 |
| textinit | s0 | 60M | 0.852 | 0.627 | — | 0.377 | 42.5 |
| textinit | s1 | 60M | 0.556 | 0.235 | ~0.53 | 0.634 | 5.50 |
| textinit | s2 | 60M | 0.578 | 0.232 | ~0.59 | 0.484 | 12.2 |

## C. Per-head correlation, full table

**Table 3 — Pearson r(attn→pos0, value-norm ratio), per-(layer,head), final checkpoint,
n=270/arm.**

| arm | r | significance |
|---|---|---|
| baseline | +0.67 | *** |
| g1gate | +0.53 | *** |
| sigmoid | −0.03 | ns |
| textinit | −0.76 | *** |
| RF | +0.43 | *** |
| pooled (n=1350) | −0.20 | *** |

## D. Per-position attention mass (seed-0, addressing the pos0-anchoring caveat)

Mean and max-head attention mass at each position, at seed-0, confirming pos0 is the
maximum-mass token by mass (not just by argmax count) in every arm:

| arm | mass@pos0 (mean) | max-head@pos0 | argmax-mass position | verdict |
|---|---|---|---|---|
| baseline | 0.06 | 0.17 | 0 | pos0 max; diffuse (no sink) |
| g1gate | 0.07 | 0.23 | 0 | pos0 max; diffuse |
| sigmoid | 0.30 | 0.66 | 0 | pos0 max; broad raw-sigmoid mass |
| textinit | 0.63 | 0.99 | 0 | pos0 max; razor spike (pos1 = 0.009) |

Seed-1/seed-2 per-position **mass** profiles (as opposed to argmax counts) were not available
locally at the time of writing (require a cloud reprobe re-dump); see Limitations §(1).

## E. References

Full citation strings and DOI/arXiv links to be verified by the author before submission
(marked with the arXiv IDs used in-text; several page/author details below are drawn from a
secondary literature-review pass and should be spot-checked against the primary sources).

- Gu, X. et al. "When Attention Sink Emerges in Language Models: An Empirical View."
  arXiv:2410.10781 (ICLR 2025).
- Queipo-de-Llano, N., Arroyo, D., Barbero, F., Dong, Y., Bronstein, M., LeCun, Y.,
  Shwartz-Ziv, R. "Attention Sinks and Compression Valleys in LLMs are Two Sides of the Same
  Coin." arXiv:2510.06477.
- Sun, M., Canziani, A., LeCun, Y., Zhu, C. "The Spike, the Sparse and the Sink: Anatomy of
  Massive Activations and Attention Sinks." arXiv:2603.05498.
- Chen, Y., Yao, Z. "Attention Sinks Induce Gradient Sinks." arXiv:2603.17771.
- Choi, S. et al. "When Sinks Help or Hurt: [Layer-wise Sink Gating in LVLMs]."
  arXiv:2604.03316.
- Luo, Y. et al. "To Sink or Not to Sink: [ViT-propagated vs. LLM-emerged sinks in LVLMs]."
  arXiv:2510.08510.
- Qiu, Z. et al. "Gated Attention for LLMs." arXiv:2505.06708 (NeurIPS 2025).
- Qiu, Z. et al. "A Unified View of Attention and Residual Sinks: Outlier-Driven Rescaling is
  Essential for Transformer Training." arXiv:2601.22966.
- Peng, J. et al. "How Attention Sinks Emerge in Language Models: An Interpretability
  Perspective." arXiv:2603.06591.
- Guo, T. et al. "Active-Dormant Attention Heads." (2024 — exact venue/arXiv ID to verify).
- Su, Y. et al. "Attention Sink in Transformers: A Survey." arXiv:2604.10098.
- Darcet, T. et al. "Vision Transformers Need Registers." (arXiv ID to verify —
  commonly cited as arXiv:2309.16588; confirm before citing).
- Sun, M. et al. "Massive Activations in Large Language Models." (2024 — original massive
  -activations paper cited by Gu et al.; exact arXiv ID to verify).
- Cancedda, N. "Spectral Filters, Dark Signals, and Attention Sinks." (2024 — cited by Gu et
  al. as the source of the massive-activation framing; exact arXiv ID to verify).

---

## Manual-check reminder (do not skip)

This draft was assembled by the Drafter agent from Auditor-approved sources
(`deliverables/session4_n3_audit.md`, `runs/rf_fresh_baseline/GATE_A_REPORT.md`,
`deliverables/preprint_readiness_audit.md`, figure-suite handoff `handoffs/handoff-director-
figure-suite-0e9a.md`) and a Researcher literature pass (`sources/researcher-related-work.md`).
It has **not** been published or submitted anywhere, and the Drafter never will. Before you
submit:

1. **Read every section against your own voice** — this is machine-drafted prose stitched
   from bullet-scaffolds; treat it as a first pass, not a final draft, even though it's
   labeled v1.
2. **Verify the appendix references** — several arXiv IDs/authors (Guo et al., Darcet et al.,
   Sun et al. "Massive Activations," Cancedda, Choi et al., Luo et al.) are flagged
   "to verify" and were not independently re-checked against the primary source by this pass.
3. **Confirm on arXiv yourself before submitting**: do the final cs.CV/cs.LG June 10–29 scoop
   browse, and make sure your first-time-submitter endorsement is sorted.
4. Figures referenced in §4 and the Appendix exist at `analysis/fig{1..6}_*.svg` — insert them
   yourself; this draft does not embed images.
