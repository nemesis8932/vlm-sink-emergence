# Appendix

## A. Extended methodology

*Probe.* The function `probe_sinks()` recomputes the decoder in eager mode, fp32, without
gradients. Every call compares its final hidden states with the model's forward pass:
the maximum absolute difference on valid positions divided by the maximum absolute
reference value must be below 0.01. Training uses fused attention, bf16 autocast and
`torch.compile`. The diagnostic excludes padding queries and query position 0.
Attention averages pool valid queries across examples. Norms are averaged before ratios,
as specified in Section 3.3.

*Optimization.* AdamW uses weight decay 0.1 and gradient clipping at 1.0. Learning rates
are $4\times10^{-4}$ for the decoder, $2\times10^{-3}$ for the projector and $10^{-4}$
for the encoder, with cosine decay and 3% warmup. Batch size is 128. The gated condition
adds a zero-initialized output gate, and textinit imports pretrained decoder weights.

*Compute.* All training and probing ran on a single NVIDIA RTX 4090 with 48 GB of
memory. The four-condition comparison and the 1B-token RF run together used
approximately 48 hours of GPU time.

*Token accounting.* Budgets count 49 image tokens plus the non-padding text tokens in each
example. A step of 128 examples therefore contributes at most 16,384 tokens and about
9.5K in the repeated-data runs. These are processed tokens, not a count of unique examples
or solely the tokens used in the text loss.

*Probe batches.* Live training trajectories use a fixed 32-example batch from the
repeated The Cauldron tail, including for RF. Saved-checkpoint reprobes provide additional
per-group and per-position arrays. The local streaming reprobe constructs a different
32-example batch with seed 0. Figure 3 uses batch-mean saved matrices, with its sigmoid
column regenerated on that streaming batch (Appendix F). Results from different probe
products are identified separately rather than treated as numerically identical.

*Validation.* The held-out image pool contains 1,024 examples. Every 500 steps, loss is
averaged over its first 512 examples in four fixed batches of 128. Repeated-data runs
also evaluate training images with fresh question–answer text (`val_seen`).
RF's seen-image loader reuses its held-out pool, so only `val_unseen` is reported.
RF ends at 0.638. Its mean held-out loss falls from 1.35 over the first ten evaluations
to 0.68 over the last ten, with a negative fitted slope in the second half.

*LLM assistance.* Large-language-model coding assistants were used to write and refactor
the training, probe and analysis code, and to edit prose. The experimental design, the
measurements, the analysis decisions and the claims are the authors' own. The released
repository keeps its commit history, including the assistant-authored commits.

## B. Extended limitations

### B.1 Position and norm aggregation

Headline measurements anchor on the first image token. Appendix E checks attention mass
across positions and reports a supplementary norm scan. Textinit's attention and norm
extrema need not coincide, and their locations vary by seed. The position-0 ratios
therefore characterize that position. Relocating the measurement to the attention maximum
does not necessarily increase the residual-norm ratio. Main norm ratios compare image
position 0 with a mixed image/text denominator, while the supplementary image-only
profiles use a different aggregation described in Appendix E.

### B.2 Gating and initialization

Our gate begins at $\sigma(0)=0.5$, halving the attention output before the output
projection. The measured difference from baseline combines this initial scale change
with subsequent learned gating. A constant half-scale control would distinguish the
two explanations. The present comparison cannot do so.

### B.3 Pretrained encoder

The SigLIP encoder is pretrained and trainable throughout. High-norm visual tokens [25]
or propagated visual sinks [11] may contribute to decoder-side signatures. RF's rising
h-ratio establishes change during multimodal training, but cannot distinguish new decoder
structure from amplification or adaptation of inherited representations. Random-encoder
and frozen-encoder controls would separate these possibilities.

### B.4 Seed variation

Textinit's h-ratio spans 5.5–42.5 at the reported checkpoints. Strong concentration and
value drain recur across seeds, but their magnitudes and spatial locations vary.
The study reports ranges and individual measurements rather than treating seed 0 as
representative of the arm's magnitude.

### B.5 RF restart and provenance

RF contains one weights-only restart near 57M tokens after an out-of-memory failure:
weights were restored and AdamW moments discarded. Recorded v-ratio and h-ratio values
agree at the shared checkpoint, and concentration is zero on both sides. The restart
remains part of the training trajectory. Main per-seed tables combine archived seed-0
summaries with independently consolidated later seeds. Figures 2 and 4 retain the
available seed-0 probe trajectories. Raw checkpoint reprobes support the per-group
analysis separately from those archived table summaries.

### B.6 Data and evaluation scope

RF reduces repetition by changing the training mixture to FineVision. The estimated
overlap with the repeated subsets is under 3% by configured dataset composition, without
image-level deduplication. Its shuffle buffer decreases from 1,500 to 500 during training
for memory reasons. Domain shift, this ordering change and the optimizer restart prevent
interpreting RF as an isolated causal test of repetition. RF's signatures are evaluated
on The Cauldron probes, so the absence result is specific to that evaluation distribution.
A matched fresh/repeated FineVision comparison and a second RF seed are direct follow-ups.

## C. Supporting figures

Figures 4 to 6 support Sections 4.1 and 4.3. Figure 4 locates concentration by layer
and query head, Figure 5 shows the per-group associations summarized in Table 3, and
Figure 6 records the attention entropy over the same runs, which falls in the two
conditions with strong concentration and does not in the others.

<figure id="figA1">
<img src="figures/fig1_layerhead_grid.svg" alt="Layer-by-head first-position attention at three checkpoints">
<figcaption><b>Figure 4: First-position attention by layer and query head.</b>
Initialization, an intermediate checkpoint near one quarter of each run's steps, and the
final probe, seed 0. Rows within each panel are layers and columns query heads. Inset
numbers are optimizer steps. Attention is row-normalized for sigmoid. All panels share
a scale saturated at 0.6. Run endpoints are those of Figure 2.</figcaption>
</figure>

<figure id="figA2">
<img src="figures/fig3_perhead_scatter.svg" alt="Attention and value-norm ratios across KV groups">
<figcaption><b>Figure 5: Concentration–value-norm associations by condition.</b>
Each point is one layer/KV-group pair, 90 per condition, at the final available seed-0
reprobe. Attention averages the group's three query heads. Its value ratio is counted
once. Pearson correlations match Table 3. The observations are descriptive and retain
dependence between groups within a layer.</figcaption>
</figure>

<figure id="figA3">
<img src="figures/fig5_entropy.svg" alt="Entropy of attention marginals over training">
<figcaption><b>Figure 6: Entropy of attention over key positions.</b>
Shannon entropy of the query-averaged key distribution, normalized by log(128), then
averaged over heads and layers, seed 0. This is marginal entropy, not the average entropy
of individual attention rows. Checkpoints are joined by lines. Initialization is placed
at 0.02M tokens for the logarithmic axis.</figcaption>
</figure>

## D. Full per-seed signature table

Table 4 provides the values summarized in Table 1. Seed-0 entries use the archived
matched-budget table, while Figure 2 uses full live trajectories, including baseline's
later 174M endpoint. A dash denotes a maximum not retained in this table's source.
It does not mean the metric is unavailable in the trajectory logs.

**Table 4: Per-seed signatures at the comparison checkpoints.**

| arm | seed | tokens | Sink^0.2 | mean attn→pos0 | max attn→pos0 | v-ratio | h-ratio |
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

The textinit h-ratio is much larger at seed 0 than at seeds 1 and 2. This variation
persists in the later supplementary norm profiles, which also show spatial differences
(Appendix E.1). Small g1gate concentration fractions indicate a few heads above 0.2.
Some seed-0 probes also cross 0.3, as shown in Figures 7 and 8.

## E. Per-position attention and norm profiles

Table 5 reports raw attention weights from the final available seed-0 checkpoint reprobes,
averaged over valid queries and the batch. Softmax weights are normalized. Sigmoid
weights are raw and unnormalized. The sigmoid row therefore describes absolute gate
weights, distinct from the normalized concentration in Table 1.

**Table 5: Raw first-position attention in checkpoint reprobes, seed 0.**

| arm | mean at pos0 | maximum head at pos0 | position with greatest mean weight |
|---|---|---|---|
| baseline | 0.059 | 0.165 | 0 |
| g1gate | 0.068 | 0.234 | 0 |
| sigmoid | 0.301 | 0.659 | 0 |
| textinit | 0.627 | 0.986 | 0 |
| RF | 0.044 | 0.094 | 1 |

### E.1 Spatial checks and image-only norm comparison

The supplementary norm scan records extrema within the first 20 image positions.
Table 6 reports that scan alongside the attention-profile checks. Dashes denote
quantities not included for those seeds. These positions are scan extrema, not proven
global extrema over all 128 positions.

**Table 6: Positions of attention maxima and norm extrema in supplementary scans.**

| run | seed | attention maximum | residual maximum | value minimum |
|---|---|---|---|---|
| baseline | s0 | 0 | 0 | 1 |
| baseline | s1 | 0 | — | — |
| g1gate | s0 | 0 | 6 | 6 |
| g1gate | s1, s2 | 0 | — | — |
| sigmoid | s0 | 0 | 14 | 13 |
| sigmoid | s1 | 0 | — | — |
| textinit | s0 | 0 | 0 | 0 |
| textinit | s1 | 0 | 1 | 5 |
| textinit | s2 | 1 | 13 | 13 |
| RF | s0 | 1 | 0 | 0 |

At RF step 64,000, the image-only residual-norm ratio is 2.516 and the value-norm ratio
0.623. These supplementary ratios first average norms over layers and, for values, KV
groups, then divide position 0 by the median across image positions 1–48. They retain the
qualitative norm asymmetry with an image-only reference, but are not numerically
interchangeable with the main layer-averaged, mixed-position ratios. The scan has no
matched initialization profile and does not independently establish the twofold growth.

For textinit seed 1, the attention maximum stays at position 0 while the residual maximum
and value minimum occur at positions 1 and 5. At seed 2, attention peaks at position 1 and
both norm extrema at position 13. Measuring at the attention maximum in the seed-2
supplementary profile gives an h-ratio of 9.46, below the position-0 ratio of 14.70.
Spatial separation therefore does not make the reported position-0 ratios lower bounds
on the norm ratio at the attention sink.

## F. Measurement and rendering details

**Figure 2 trajectories.** Moving-average windows contain five probes horizontally and
nine vertically. Faint points show unsmoothed values. Open circles and final diamonds
use raw endpoints. All conditions use seed 0 and their full available trajectories.
The horizontal maximum and the thresholded fraction in Table 1 are different summaries
of the same head-level attention scores.

**Group correlations.** Table 3 uses the final available per-head reprobes: baseline
step 18,287; g1gate 10,786; sigmoid 10,664; textinit 6,244; RF 64,000. The RF reprobe
precedes its final live probe. Value ratios are retained per KV group, with attention
averaged over the group's query heads. Uncollapsed query-head correlations are
+0.67, +0.53, −0.03, −0.76 and +0.43 respectively. Collapsing preserves their signs.
Neither aggregation removes layer dependence.

**Raw and normalized sigmoid weights.** Gu et al. [6] also row-normalize
proxy weights when measuring sinks in models trained with unnormalized sigmoid attention.
Our normalized diagnostic is consistent with that distinction between measurement and
forward computation. In the saved sigmoid reprobe, head L7H3 has mean normalized
first-position attention 0.873 at step 10,664, raw first-position weight 0.065, and
mean total raw row weight 0.110. At step 250, the corresponding raw quantities are
0.309 and 6.44. The growing relative concentration thus accompanies a shrinking
absolute weight budget. The ratio of mean raw weights, about 0.59 at the endpoint,
differs from the mean normalized attention, 0.873, because averaging and division do
not commute.

**Figure 3 batches and selection.** The panels show batch-mean matrices over 128 positions:
49 image positions followed by left-padded text. They do not correspond to a single
prompt. Each matrix is normalized by row after batch averaging, whereas the headline
probe averages normalized weights over valid queries. Scalar headline measurements
are therefore not overlaid on the qualitative matrices.

For baseline, g1gate and textinit, we choose the higher final-checkpoint first-position
share among the two heads retained in the original matrix dump, then hold that head
fixed across displayed checkpoints. These are L1H1, L5H5 and L18H8 respectively,
using zero-based indices. The sigmoid panels use L7H3, selected using normalized
attention in the checkpoint reprobe and regenerated from a streaming batch. The exact
decoded text strings were not retained with the matrix archive.

Bright off-position-0 cells show query-specific attention to other keys. For example,
the final sigmoid map has strong cells in the text-to-text region. Such isolated
cells do not by themselves establish a persistent sink or identify the semantic role
of the attended tokens. Padding and batch averaging also affect the qualitative
display. The common power-law color scale uses exponent 0.5 and saturates at 0.5.

## G. Mechanistic hypotheses

The following explanations motivate future controls. They are not established by the
signature measurements.

An output gate could reduce the need to suppress a value vector, consistent with the
milder value drain in g1gate. A scale-matched ungated control is needed to distinguish
that explanation from initial output scaling. Under sigmoid attention, removing the
sum-to-one constraint permits a small total attention update. Testing how this relates
to value amplification requires measuring effective updates as well as norms.
Text initialization may transfer a first-position circuit to the visual prefix.
Ablating that circuit before alignment would test its role in the inherited profile.

## H. Metric provenance

The main per-seed table, live trajectories and checkpoint reprobes are distinct products.
Mean attention, maximum attention and the fraction above a threshold are reported under
separate names. Positional attention profiles use weights over the full sequence,
without renormalizing a displayed slice. Position-0 norm ratios remain labeled by their
anchor even when supplementary scans find larger norms elsewhere.

## I. Threshold ordering and spatial dissociation

<figure id="figA4">
<img src="figures/fig4_leadlag_top.svg" alt="First threshold crossings by condition">
<figcaption><b>Figure 7: First threshold crossings during training.</b>
Filled markers indicate the first recorded crossing of h-ratio &gt; 2, v-ratio &lt; 0.8,
or maximum attention &gt; 0.3. Hollow markers mark conditions with no observed crossing
by the end of the track. Initialization is placed at 0.04M tokens on the log axis.
Crossing times are interval-censored by the 100-step probe cadence. All runs use seed 0.</figcaption>
</figure>

<figure id="figA5">
<img src="figures/figA4_birthmap.svg" alt="First attention-threshold crossing by layer and head">
<figcaption><b>Figure 8: First concentration crossing by layer and query head.</b>
Colors show the first step with mean attention to position 0 above 0.3. White cells
never cross during the recorded trajectory. The fractions crossing at least once are
1% for g1gate, 89% for sigmoid and 87% for textinit. Baseline and RF have no crossing
heads. Layers increase downward and query-head indices increase to the right.</figcaption>
</figure>

**Ordering.** In baseline, residual-norm ratio first exceeds 2 at 3.82M tokens and
value-norm ratio falls below 0.8 at 11.45M, with no concentration crossing.
G1gate crosses the norm thresholds at 5.72M and 20.98M, then crosses concentration at
55.32M in a small number of heads. Sigmoid crosses concentration at 5.72M without
crossing either norm threshold. Textinit starts above the residual threshold and below
the value threshold, then crosses concentration by 0.95M tokens.
RF crosses the norm thresholds at 14.44M and 51.64M without crossing concentration.
These are first observed events, not guarantees of sustained crossings.

The norm extrema also separate spatially in textinit (Appendix E.1). Together with
the temporal patterns, this supports tracking each signature with its own threshold,
position and aggregation, rather than inferring one from another.
