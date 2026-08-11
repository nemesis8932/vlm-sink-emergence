# 3. Results

<figure id="fig1">
<img src="figures/fig2_phase_portrait.svg" alt="Decoupling phase portrait">
<figcaption><b>Figure 1: Decoupling phase-portrait.</b> Trajectories of every arm in
concentration-against-value-norm space (left) and against the residual-norm ratio (right,
log scale). Circles mark initialization, diamonds the final checkpoint: about 100M tokens
for <em>baseline</em>, <em>g1gate</em> and <em>sigmoid</em>, 60M for <em>textinit</em>, 1B
for <em>RF</em>. Seed 0 unless labelled. <b>The horizontal axis differs from the tables:</b>
it is the <em>maximum</em> attention→pos0 over heads, where Table 1 reports
Sink<sup>ε</sup><sub>1</sub>, the <em>fraction of heads above a threshold</em>, so an arm
can sit off zero here with Sink<sup>ε</sup> = 0.000. Paths are smoothed, end markers use raw
values (Appendix D). Were the three signatures one phenomenon, these trajectories would move
along one direction.</figcaption>
</figure>

## 3.1 Four training levers reach four different signature corners

Table 1 collapses the seeds into per-arm ranges, each arm read at its final matched
checkpoint: about 100M tokens for baseline, g1gate and sigmoid, and 60M for *textinit*,
whose signatures have plateaued by then. Appendix B gives the per-seed table.
**No two arms share a signature triple.**

**Table 1 — the four corners (n = 2–3 seeds per arm).** Concentration is Sink^0.2, the
fraction of the 270 (layer, query-head) pairs whose mean attention→pos0 exceeds 0.2. The
v-ratio rests on 90 (layer, KV-head) value projections and the h-ratio on 30 layers (§2).

| arm | concentration | value-norm | massive-activation proxy |
|---|---|---|---|
| baseline | absent (0.000) | mild drain (0.69–0.72) | moderate (1.7–2.2) |
| g1gate | near-absent (0.004–0.011) | **milder drain** (0.81–0.85) | moderate (1.7–2.2) |
| sigmoid | strong (0.76–0.83) | **amplified** (1.48–1.60) | no strong asymmetry (1.1–1.3) |
| textinit | strong (0.56–0.85) | strong drain (0.38–0.63) | extreme (5.5–42.5) |

The value-norm axis alone takes three qualitatively different directions: drained hard,
drained mildly, or amplified above 1. No arm leaves it unchanged.

The corners separate pairwise on single axes. *g1gate* differs from *baseline* on the
value-norm axis alone, since concentration is absent or near-absent and the residual-norm
ratio moderate in both, while the gate makes the baseline's mild drain milder still, a
15–19% drain rather than none. *sigmoid* and *textinit* both reach strong concentration and
then part on the *direction* of the value-norm move and on the residual-norm ratio. Each arm
shows a distinct intervention-associated profile. The arms are not a factorial design, so we
do not attribute these profiles to isolated causal effects of single levers. The
trajectories in Figure 1 separate early and do not share one origin: *textinit* starts from
a pretrained text decoder, with an inherited sink already in place.

Reproducibility differs by arm. *g1gate* is tightest, at Sink^0.2 of 0.004, 0.011 and 0.0037
across three seeds. The corner of *textinit* repeats across seeds and its magnitudes do not:
seed 0 sits far above the other two on every signature at once, at h-ratio 42.5 against
5.5–12.2, partly because at seeds 1 and 2 its peak signatures move off position 0, where our
metrics anchor (§3.4, Appendix B). We therefore report the massive-activation proxy of
textinit as a **range (5.5–42.5×) with a median near 12×** throughout, and treat the corner
as the reproducible claim rather than any magnitude.

## 3.2 The massive-activation proxy grows across 1B low-repetition tokens while concentration stays at zero

The four-arm comparison reuses a 146K-image pool, so a reader could read its "concentration
never emerges" result as an overfitting artifact. The RF arm addresses that: the identical
*baseline* recipe on a fresh FineVision stream to one billion tokens, at **2.39 effective
visual epochs**. Examples still repeat, about 2.4 times on average, so this is a
**low-repetition** run rather than a repetition-free one. Its held-out loss falls through
the full billion tokens and never turns upward (§2, §5).

Concentration never arrives. **Sink^0.3 = 0.000 across the entire 0 → 1B run**, with no
head of the 270 crossing the threshold at any of about 700 probes. The massive-activation
proxy, in the same run, keeps growing far past warmup.

**Table 2 — RF (fresh stream, 2.39 visual epochs, n = 1 seed), init → 1B tokens.**

| signature | @ init | @ 1B | net |
|---|---|---|---|
| Sink^0.3 (concentration) | 0.000 | 0.000 | flat zero, entire run |
| max attn→pos0 | 0.056 | 0.098 | ≈flat, far below threshold |
| h-ratio (massive-activation proxy) | 1.43 | **3.22** | **≈2.3×**, positive long-horizon trend |
| v-ratio (value-norm) | 1.00 | 0.69 | net drain, non-monotone |

Warmup does not explain the rise. The h-ratio climbs from 2.40 at about 57M tokens to 3.22
at 1B, long after the 3% warmup window closes, though not monotonically at probe resolution,
so we report it as a positive long-horizon trend. In Figure 1, at right, the violet
trajectory climbs and never moves rightward. The v-ratio ends below its initialization
value, but about 75% of that drop happens in the first 57M tokens and it then recovers part
of it, so we report it as supporting context rather than ongoing emergence. **The massive-activation proxy grows about 2.3× across a full billion tokens of
multimodal training while attention concentration never leaves zero (n = 1 seed, §5).**

## 3.3 No consistent-sign head-level relationship between concentration and value-norm

A tight coupling between concentration and value-drain should at minimum hold the sign of
their per-head correlation constant across regimes. It does not. Table 3 reports the Pearson
r between attention→pos0 and value-norm ratio at each arm's final checkpoint, over the 90
(layer, KV-group) observations: a value vector is shared by 3 query heads, so we average a
group's query-head attention rather than triplicate its value observation (§2). Appendix
Fig. A2 plots the scatter.

**Table 3 — r(attn→pos0, v-ratio), final checkpoint. Descriptive only, no p-values.**

| baseline | g1gate | sigmoid | textinit | RF | pooled |
|---|---|---|---|---|---|
| +0.76 | +0.57 | −0.04 | −0.79 | +0.51 | −0.20 |

These correlations are **descriptive statistics, not inferential ones**, and we deliberately
report no significance tests: heads within a layer are not independent, and each arm rests
on a single seed here, so a p-value would mislead. Collapsing to the 90 KV groups removes
the pseudoreplication a per-query-head reading would introduce and leaves the picture
unchanged (Appendix D).

The pattern is about **sign**, which pseudoreplication does not manufacture. Heads that
attend more to position 0 have *larger* value norms in the baseline regime and *smaller*
ones under text initialization, and the pooled correlation is weak only because arms of
opposite sign cancel. We therefore claim no consistent-sign relationship across arms. We do
not claim the absence of a coupling law: a fixed coupling would show one sign everywhere,
and it does not.

## 3.4 Ordering: norm signatures lead, and concentration is late, mirrored, or absent

<figure id="fig2">
<img src="figures/fig4_birth_leadlag.svg" alt="Sink birth and lead-lag ordering">
<figcaption><b>Figure 2: Sink birth and ordering.</b> Seed 0 throughout. <em>Top:</em>
time-to-event tracks spanning the tokens over which we observed each arm (60M to 1B). A
filled marker is the first probe at which a signature crossed its per-head threshold
(h&gt;2, v&lt;0.8, attn→pos0&gt;0.3). A hollow marker at a track's end means it never
crossed. <b>Crossing times are interval-censored</b> at the 100-step probe cadence, so two
signatures crossing within one interval should not be read as ordered. <em>Bottom:</em>
birth-maps, the step at which each (layer, query-head) pair first crosses the concentration
threshold. No baseline and no RF head ever crosses. 1% of g1gate heads do, against 89% for
sigmoid and 87% for textinit.</figcaption>
</figure>

Dense probing lets us timestamp each signature's arrival (Figure 2), and the ordering stays
consistent inside each arm family. In the softmax arms with randomly initialized decoders,
*baseline* and *g1gate* and *RF*, the norm signatures come early: the residual-norm ratio
crosses within the first few to about 15M tokens and value-drain follows within about 50M.
Concentration never comes. No baseline or RF head ever crosses it, and g1gate reaches 1% of
heads, a single-layer blip late in training. *sigmoid* mirrors that pattern, with
concentration crossing at about 6M tokens and 89% of heads eventually, while neither norm
signature ever crosses. *textinit* inherits its norm signatures rather than growing them:
value-drain and an elevated residual-norm ratio are both present at 0 tokens, imported with
the pretrained text-LM weights. Concentration is *not* inherited the same way, since
Sink^0.3 is 0.000 at step 0 and crosses before 1M tokens. Even in the arm that imports the
most sink structure, the norm signatures precede concentration.

<figure id="fig3">
<img src="figures/fig6_sink_stripe.svg" alt="Query-by-key attention maps, top sink head per arm">
<figcaption><b>Figure 3: The sink stripe.</b> Query &times; key attention of each arm's top
sink head, row-normalized, seed 0. Panels carry their seed, step and token count. Top row is
early training (step 250, about 2.4M tokens), bottom row the final checkpoint. The cyan line
marks key = pos0, the dotted line the image-to-text boundary. The pos0 stripe is
<em>absent</em> in baseline throughout and strong in textinit (attn→pos0 = 0.62 at its final
checkpoint). The rightmost panel shows it <em>already present at step 0</em> in textinit,
inherited from the text language model. The <em>sigmoid</em> column uses head <b>L7H3</b>,
that arm's top sink head on the fixed probe batch. Its heat maps come from an auxiliary
streaming batch, while every printed sigmoid number comes from the fixed probe batch
(Appendix D).</figcaption>
</figure>

Figure 3 shows the same result at the level of a single head. The vertical stripe at
key = pos0 marks every query attending to the first image token. It is absent in *baseline*
at both checkpoints and strong in *textinit*, and the rightmost panel carries the most
weight: **the stripe is already there at step 0 in textinit**, imported with the text-LM
weights before the model has seen one image.

The *sigmoid* arm needs one measurement note, because row-normalization changes the object
being measured and Gu et al. [6] state their result for *unnormalized* sigmoid attention.
The concentration this arm develops is a **relative** reallocation of a shrinking gate
budget onto position 0, not the growth of a large absolute mass there (Appendix D). We
report the row-normalized view in the tables so the arms stay comparable.

Attention-entropy collapse, the text-LM literature's usual correlate of concentration,
separates the same way: only *sigmoid* and *textinit* collapse (Appendix Fig. A3). It tracks
the concentration axis, not the norm axes.

**The signatures also dissociate in position.** A per-position scan across all three
*textinit* seeds shows they need not share a token. At seed 0 they coincide at position 0.
At seed 1 the attention maximum stays there while the residual peak moves to pos1 and the
value minimum to pos5. At seed 2 they separate furthest, attention at **pos1** and both norm
extrema at **pos13**. The arms with randomly initialized decoders behave differently:
position 0 stays the maximum-mass token in *baseline*, *g1gate* and *sigmoid* at every seed
scanned, and in *RF* the attention argmax sits at pos1 with mass 0.100 against 0.083 at
position 0, a diffuse profile rather than a sink (Appendix C).

We report this as a supporting observation, not a second headline. It has one consequence
for measurement: because all three metrics anchor on position 0, the *textinit* magnitudes
at seeds 1 and 2 are read at a token that is no longer the peak and therefore **understate**
the arm's true peaks, a further reason to report that arm as a range and a median (§3.1,
§5). The anchoring holds for the randomly initialized decoders, which carry the central
negative result. Appendix E sets out, as untested hypotheses, how each lever might move a
different axis.
