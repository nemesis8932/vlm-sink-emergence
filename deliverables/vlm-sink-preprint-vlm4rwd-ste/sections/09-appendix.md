# Appendix

## A. Extended methodology

*Probe.* The function `probe_sinks()` re-walks the decoder from the live module weights in
eager mode, fp32, no gradients, independent of the training path (fused SDPA kernels,
autocast, `torch.compile`). Every call validates its hidden states against the real forward
pass to a relative error below 10<sup>−2</sup>, so the probe cannot drift from what the
trained model computes. The probe batch is fixed across all runs and seeds, which keeps every
number comparable, and probes fire every 100 optimizer steps.

*Recipe.* AdamW with weight decay 0.1, following [6], gradient clip 1.0, cosine schedule
with 3% warmup. Arms in a comparison differ only in the lever under test.

*Optimization.* The learning rates are 4 &times; 10<sup>&minus;4</sup> for the language
model, 2 &times; 10<sup>&minus;3</sup> for the projector, and 10<sup>&minus;4</sup> for the
vision encoder. We use bf16 autocast, `torch.compile`, and a batch size of 128.


*Token accounting.* One token is one image token or one non-padding text token. A full
sequence contributes 49 image tokens plus its non-padding text tokens, so a step of batch
128 contributes at most 16,384 tokens and about 9.5K on average. All "M tokens" figures in
this paper use that definition.

*Probe batch.* The fixed probe batch holds n = 32 samples, drawn with `random.seed(0)` from
the repeated `the_cauldron` tail. We label this probe version `v1-repeatedtail-32`. RF uses
the same probe batch as the repeated arms, so its signatures stay comparable to theirs even
though its training stream differs (§5).

*Validation.* A 1,024-example held-out pool, evaluated every 500 steps. Each estimate is the
mean over the first 512 examples of that pool, in four batches of 128 in fixed order. We
report `val_unseen`, which holds out images. For the repeated arms we also log `val_seen`,
which re-uses training images with fresh question–answer text. The RF run has no distinct
seen split. At 2.39 visual epochs its `val_seen` loader re-uses the held-out pool, so we
report only `val_unseen` for RF. At the matched 100M-token checkpoint the held-out
losses are 1.182 for *baseline*, 1.133 for *g1gate*, 1.206 for *sigmoid* and 0.877 for
*textinit*, and RF reaches 0.638 at 1B tokens. The seed-0 archive values for that
checkpoint are 1.161 for *baseline*, 1.138 for *g1gate*, 1.108 for *sigmoid*, and 0.832 for
*textinit*, so the first set is seed 1 or 2. For RF the held-out loss goes from 1.35
over the first ten evaluations to 0.68 over the last ten, and the fitted slope over the
second half of the run stays negative.

*Aggregation.* Each per-layer or per-head quantity is first averaged over the probe batch
and over valid query positions, then aggregated by the formulas of §3.

## B. Extended limitations

<!-- filled in the limitations chunk -->

## C. Supporting figures

<figure id="figA1">
<img src="figures/fig1_layerhead_grid.svg" alt="Per-(layer,head) attention to pos0 over training">
<figcaption><b>Figure A1: Where concentration lives in the network.</b> Mean attention to
pos0 for each of the 270 (layer, query-head) pairs through training, row-normalized, seed 0
for every arm, run to each arm's final checkpoint (174M tokens baseline, 103M g1gate, 102M
sigmoid, 60M textinit, 1B RF). baseline and RF stay cold throughout. textinit is hot from
initialization. sigmoid lights up a band of mid-network layers.</figcaption>
</figure>

<figure id="figA2">
<img src="figures/fig3_perhead_scatter.svg" alt="Per-head concentration vs value-norm scatter">
<figcaption><b>Figure A2: No consistent-sign head-level relationship.</b> Concentration
against value-norm ratio for each (layer, KV-group) pair at each arm's final checkpoint,
seed 0, at n = 90 pairs per arm. Grouping this way avoids triplicating each value-norm
observation across the 3 query heads that share it, so these correlations are descriptive
and we report no p-values (&sect;3.3). The sign of the correlation flips by arm, from +0.76
in baseline to &minus;0.79 in textinit (Table 3). The pooled cloud is weak, at &minus;0.20,
only because arms of opposite sign cancel. Do not read it as an uncorrelated cloud. Most
individual arms show a strong |r|.</figcaption>
</figure>

<figure id="figA3">
<img src="figures/fig5_entropy.svg" alt="Attention entropy over training by arm">
<figcaption><b>Figure A3: Entropy collapse tracks concentration only.</b> Mean attention
entropy through training, seed 0, to each arm's final checkpoint. Only sigmoid and textinit
collapse. baseline, g1gate, and RF stay flat. The entropy-collapse correlate from the
text-LM literature therefore follows the concentration axis, not the norm axes.</figcaption>
</figure>

## D. Full per-seed signature table

This table gives the per-seed values behind the collapsed ranges of Table 1, and adds the
maximum attention→pos0 where we have it. We logged that metric for seeds 1 and 2 only,
because the seed-0 archive summary does not include it. We trust the seed-0 values from a
checksummed archive rather than re-derive them first-hand (§5). Each arm is read at its
final matched checkpoint. All three signatures of *textinit* have plateaued at 60M tokens,
and its h-ratio changes by less than 10% from 60M to 100M in the seeds that continued.

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

*baseline* and *sigmoid* are tight across both of their seeds. *textinit* is the loose arm.
Seed 0 sits far above the other two on every signature at once, at Sink 0.85 against
0.56–0.58, mean attn→pos0 0.63 against 0.23, and h-ratio 42.5 against 5.5–12.2. The h-ratio
has plateaued by 60–100M tokens in both lower seeds, so the spread is genuine seed
sensitivity rather than an unconverged transient. Part of it is positional (Appendix E.2).

A note on g1gate. Both seeds with max-attention data show one head at about 0.21–0.22
maximum attention→pos0. That single head does clear the strict ε = 0.2 threshold, which is
why the arm's Sink^0.2 is small but not exactly zero, and no head comes close to the ε = 0.3
default of [6]. The arm mean meanwhile stays near 0.07 and Sink^0.2 stays far below 0.05.
That pattern repeats across seeds.

## E. Per-position attention mass (seed 0)

This table gives the mean and max-head attention mass by position at seed 0. It confirms
that position 0 is the maximum-mass token in every arm, by mass rather than by argmax count
alone. It is the direct check behind the position-0 anchoring defense of §5.

| arm | mass@pos0 (mean) | max-head@pos0 | argmax-mass position | reading |
|---|---|---|---|---|
| baseline | 0.06 | 0.17 | 0 | pos0 max; diffuse (no sink) |
| g1gate | 0.07 | 0.23 | 0 | pos0 max; diffuse |
| sigmoid | 0.30 | 0.66 | 0 | pos0 max; broad raw-sigmoid mass |
| textinit | 0.63 | 0.99 | 0 | pos0 max; razor spike (pos1 = 0.009) |

### E.2 Per-position scan at the remaining seeds, and for RF

We re-dumped per-position attention mass at the other seeds and for RF, and per-position
residual and value norms for all three *textinit* seeds. The table gives the position at
which each quantity peaks (for value norms, the position at which the norm is smallest).

| run | seed | attention-mass argmax | residual-norm peak | value-norm minimum | reading |
|---|---|---|---|---|---|
| baseline | s0, s1 | 0 | — | — | pos0 max |
| g1gate | s0, s1, s2 | 0 | — | — | pos0 max |
| sigmoid | s0, s1 | 0 | — | — | pos0 max |
| textinit | s0 | 0 | 0 | 0 | all three coincide |
| textinit | s1 | 0 | 1 | 5 | norms move off pos0 |
| textinit | s2 | **1** | **13** | **13** | attention separates from coincident norm extrema |
| RF | s0 | 1 | — | — | mass 0.053 vs 0.044 at pos0, diffuse, not a sink |

Two readings follow. Position 0 remains the maximum-mass token in every arm with a randomly
initialized decoder, so the pos0 anchoring holds where the central negative result lives.
In *textinit* the three signatures need not share a token, which is the positional
dissociation of Appendix I and the reason the pos0-anchored textinit magnitudes at seeds
1 and 2 understate that arm's peaks.

## F. Measurement and rendering details

**Figure 1 smoothing.** Paths are smoothed with a moving average, 5 points on the horizontal
axis and 9 on the vertical. The faint dots behind each path are the unsmoothed per-probe
values, and the initialization and final markers use raw values.

**Correlations at the uncollapsed 270 pairs.** Table 3 collapses to the 90 (layer, KV-group)
observations. At the uncollapsed 270 (layer, query-head) pairs the same quantities read
+0.67 for baseline, +0.53 for g1gate, −0.03 for sigmoid, −0.76 for textinit, and +0.43 for
RF. Every sign and every ordering survives the collapse.

**Raw against row-normalized sigmoid attention.** Row-normalization changes the object being
measured, and Gu et al. [6] state their result for *unnormalized* sigmoid attention. Head
L7H3 of the sigmoid arm sends 0.873 of its row-normalized attention to position 0 at the
final checkpoint, which is the arm maximum. Its raw gate mass to position 0 is 0.065, and
the raw mass summed over all keys in that row is 0.110. The row does not sum to one, and
pos0 takes about 59% of what little gate mass the head opens at all. Early in training the
same head shows the opposite picture: raw pos0 mass 0.309 against a raw row sum of 6.44, so
under 5% of a very large gate budget. The concentration this arm develops is therefore a
relative reallocation of a shrinking gate budget onto position 0, not the growth of a large
absolute mass there (Appendix I). Raw and row-normalized values here come from the same fixed
probe batch as every other number in this paper, masked to valid query positions.

**Sigmoid heat-map provenance.** The sigmoid column of Figure 2 uses head L7H3 (0-indexed),
the arm's top sink head on the fixed probe batch. An earlier dump selected heads by raw,
unnormalized gate score, picked different heads, and understated the arm. The sigmoid heat
maps are re-rendered from an auxiliary streaming batch, because the saved matrices covered
the superseded heads. Every printed sigmoid number, in the text and in the tables, comes
from the fixed probe batch. The pos0 share printed on each panel is computed over valid
query positions, the same convention as the tables.

## G. Interpretation (untested hypotheses)

This appendix is **speculative**. Nothing in it is tested by our experiments, and none of it
is a claim of this paper. We include it because a reader is entitled to ask *why* the
signatures come apart, and because these hypotheses are cheap to state and testable later.

Gu et al. [6] account for the sink as a key bias. Softmax must distribute a full unit of
attention mass per row, and a head with nothing informative to retrieve parks the surplus on
a token whose value contributes little. If that account is right, each of our levers touches
a different part of it, which would explain why each moves a different axis.

- *g1gate.* An output gate can suppress whatever the attended position injects into the
  residual stream. A model with that gate may be able to *afford* concentration, because
  concentration no longer forces a matching change in the value path. That would be
  consistent with a gate whose measured effect here is on the value-norm axis rather than
  the concentration axis. Fesser et al. [23] make a compatible argument from another
  direction. They distinguish two kinds of sink. An "adaptive nop" head suppresses its own
  update by routing attention to a token whose value contributes nothing, and a negligible
  value norm identifies it. A broadcast head instead spreads global information. They note
  that gating implicitly assumes the nop mechanism. If that is
  right, a gate should act on the value-norm axis first, which is where our gated arm
  differs from baseline.
- *sigmoid.* Removing the softmax removes the sum-to-one constraint itself. A head with
  nothing to retrieve can simply attend weakly to everything, with no surplus to park. On
  this reading the value amplification we observe is what the value path does when it is no
  longer compensating for a forced allocation.
- *textinit.* A pretrained text decoder arrives with sink machinery already built. Our
  observation that its norm signatures are present at step 0 while concentration crosses
  later would then reflect import of structure followed by re-targeting onto a visual
  prefix, rather than formation from scratch.

Each of these is a hypothesis about mechanism. Testing them needs interventions we did not
run: gate-scale sweeps that separate gating from the half-scale confound of §3, sigmoid runs
at matched effective attention temperature, and text-initialized runs with the sink
machinery ablated before alignment.

## H. Notes on metric hygiene

We record three corrections that we made during the audit of the numbers in this paper.
First, an earlier internal consolidation mixed two metrics in one column, mean and max
attention→pos0, and thereby manufactured an apparent seed anomaly in *g1gate*. Re-derived
like-for-like, the anomaly disappears, and g1gate becomes the most reproducible arm.
Second, an earlier draft claimed that the residual-norm peak of textinit "stays pinned at
pos0." We checked that claim against the norm dump. It is wrong. The ‖h‖ peak sits at
pos0, pos1 or pos13, depending on the seed (Appendix E.2). That is the positional
dissociation reported in Appendix I, which states the corrected form.
Third, the per-position script normalized the profile over a 20-position display slice
rather than the full 128, which inflated the reported RF masses to 0.100 and 0.083. The
true full-sequence values are **0.053 at pos1 and 0.044 at pos0**. The diffuse-profile
reading is unchanged, and the softmax-arm rows of Appendix E were unaffected, because those
profiles already sum to one over the full sequence.

## I. Ordering, the sigmoid measurement note, and positional dissociation

This appendix holds the detail behind the ordering paragraph of §4.1.

<figure id="figA4">
<img src="figures/fig4_leadlag_top.svg" alt="Sink lead-lag ordering">
<figcaption><b>Figure A4: When each signature first crosses threshold.</b> Seed 0 throughout.
Time-to-event tracks spanning the tokens over which we observed each arm (60M to 1B). A
filled marker is the first probe at which a signature crossed its threshold
(h&gt;2, v&lt;0.8, attn→pos0&gt;0.3). A hollow marker at a track's end means it never crossed. <b>Crossing times are interval-
censored</b> at the 100-step probe cadence (head-level map in Fig. A5).</figcaption>
</figure>

<figure id="figA5">
<img src="figures/figA4_birthmap.svg" alt="Birth-maps: step of first concentration crossing per head">
<figcaption><b>Figure A5: Birth-maps.</b> The step at which each (layer, query-head) pair
first crosses the concentration threshold (attn&rarr;pos0 &gt; 0.3), seed 0, for the three
arms in which any head crosses. No <em>baseline</em> and no <em>RF</em> head ever crosses.
1% of <em>g1gate</em> heads do, against 89% for <em>sigmoid</em> and 87% for
<em>textinit</em>. This is the head-level view behind Fig. A4.</figcaption>
</figure>

**Timings.** In the softmax arms with randomly initialized decoders, *baseline* and *g1gate*
and *RF*, the residual-norm ratio crosses its threshold within the first few to about 15M
tokens and value-drain follows within about 50M. Concentration never comes. No baseline or
RF head ever crosses it, and g1gate reaches 1% of heads, a single-layer blip late in
training. *sigmoid* mirrors that pattern, with concentration crossing at about 6M tokens and
89% of heads eventually, against 87% for *textinit*, while neither *sigmoid* norm signature
ever crosses (Fig. A5). *textinit* inherits its
norm signatures rather than growing them. Value-drain and an elevated residual-norm ratio
are both present at 0 tokens, imported with the pretrained text-LM weights. Concentration is
*not* inherited the same way, since Sink^0.3 is 0.000 at step 0 and crosses before 1M
tokens. Even in the arm that imports the most sink structure, the norm signatures precede
concentration. Crossing times are interval-censored at the 100-step probe cadence, so two
signatures that cross within one interval should not be read as ordered.

**The sigmoid arm needs one measurement note**, because row-normalization changes the object
being measured and Gu et al. [6] state their result for *unnormalized* sigmoid attention.
The concentration this arm develops is a **relative** reallocation of a shrinking gate budget
onto position 0, not the growth of a large absolute mass there (Appendix F). We report the
row-normalized view in the tables so the arms stay comparable.

**Entropy collapse.** Attention-entropy collapse, the text-LM literature's usual correlate of
concentration, separates the same way: only *sigmoid* and *textinit* collapse (Fig. A3). It
tracks the concentration axis, not the norm axes.

**The signatures also dissociate in position.** A per-position scan across all three
*textinit* seeds shows they need not share a token. At seed 0 they coincide at position 0.
At seed 1 the attention maximum stays there while the residual peak moves to pos1 and the
value minimum to pos5. At seed 2 they separate furthest, attention at **pos1** and both norm
extrema at **pos13**. The arms with randomly initialized decoders behave differently:
position 0 stays the maximum-mass token in *baseline*, *g1gate* and *sigmoid* at every seed
scanned, and in *RF* the attention argmax sits at pos1 with mass 0.053 against 0.044 at
position 0, a diffuse profile rather than a sink (Appendix E).

We report this as a supporting observation, not a second headline. It has one consequence for
measurement. All three metrics anchor on position 0, so the *textinit* magnitudes at seeds 1
and 2 are read at a token that is no longer the peak and therefore **understate** the arm's
true peaks, a further reason to report that arm as a range and a median (§4.1, §5).
The anchoring holds for the randomly initialized decoders, which carry the central negative
result.
