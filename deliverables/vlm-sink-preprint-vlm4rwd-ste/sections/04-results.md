# 3. Results

<figure id="fig1">
<img src="figures/fig2_phase_portrait.svg" alt="Decoupling phase portrait">
<figcaption><b>Figure 1: Decoupling phase-portrait.</b> Training trajectories of every arm
in concentration-against-value-norm space, at left, and concentration against the
residual-norm ratio, at right, on a log scale. Circles mark initialization. Diamonds mark
the final checkpoint, at about 100M tokens for <em>baseline</em>, <em>g1gate</em> and
<em>sigmoid</em>, 60M for <em>textinit</em>, and 1B for <em>RF</em>; seed 0 unless a seed is
labelled. <b>Note the metric difference from the tables:</b> the horizontal axis is the
<em>maximum</em> attention→pos0 over heads, a continuous quantity that keeps a trajectory
visible, whereas Table 1 reports Sink<sup>ε</sup><sub>1</sub>, the <em>fraction of query
heads above a threshold</em>. An arm can therefore sit at a non-zero position on this axis
while its Sink<sup>ε</sup> stays 0.000. Paths are smoothed (Appendix D); faint dots are the
unsmoothed per-probe values, and the end markers use raw values. Each lever drives the
signatures to a different corner, and every combination of directions occurs. If the three
signatures were one phenomenon, these trajectories would move together along one
direction.</figcaption>
</figure>

## 3.1 Four training levers reach four different signature corners

Table 1 collapses the seeds into per-arm ranges, over two seeds for *baseline* and
*sigmoid* and three for *g1gate* and *textinit*. Appendix B gives the full per-seed table.
Each arm is read at its final matched checkpoint: about 100M tokens for baseline, g1gate,
and sigmoid, and 60M tokens for *textinit*, whose signatures have plateaued by then.
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

The value-norm axis alone takes three qualitatively different directions. The lever drains
it hard, drains it only mildly, or amplifies it above 1. No arm leaves it unchanged.

The corners separate pairwise on single axes. *g1gate* differs from *baseline* on the
value-norm axis alone: concentration is absent or near-absent and the residual-norm ratio
moderate in both, while the gate makes the baseline's mild value-drain milder still, from
0.69–0.72 to 0.81–0.85, a 15–19% drain rather than none. *sigmoid* and *textinit* both reach
strong concentration and then part on two axes, the *direction* of the value-norm move and
the residual-norm ratio. Each arm shows a distinct intervention-associated profile. The four
arms are not a factorial design, so we do not attribute these profiles to isolated causal
effects of single levers. Figure 1 shows the trajectories separating early and ending in
four corners. They do not share one origin: *textinit* starts from a pretrained text decoder
and therefore begins with an inherited sink in place.

Reproducibility differs by arm. *g1gate* is the tightest, at Sink^0.2 of 0.004, 0.011, and
0.0037 across three seeds. The corner of *textinit* repeats across seeds, and its
magnitudes do not: seed 0 sits far above the other two on every signature at once, at
h-ratio 42.5 against 5.5–12.2 (Appendix B). Part of that spread is positional, because at
seeds 1 and 2 the arm's peak signatures move off position 0, where our metrics anchor
(§3.4). We therefore report the massive-activation proxy of textinit as a **range
(5.5–42.5×) with a median near 12×** everywhere in this paper, and we treat the corner as
the reproducible claim rather than any particular magnitude.

## 3.2 The massive-activation proxy grows across 1B low-repetition tokens while concentration stays at zero

The four-arm comparison reuses a 146K-image pool, so a reader could read its "concentration
never emerges" result as an overfitting artifact. The RF arm addresses that: the identical
*baseline* recipe on a fresh FineVision stream to one billion tokens, at **2.39 effective
visual epochs**. Examples still repeat, about 2.4 times on average, so this is a
**low-repetition** run rather than a repetition-free one. Its held-out loss falls through
the full billion tokens and never turns upward, so nothing in the run indicates overfitting.
RF has no distinct seen split, so we cannot repeat for it the `val_seen` / `val_unseen`
comparison that exposes memorization in the repeated arms (§2, §5).

Concentration never arrives. **Sink^0.3 = 0.000 across the entire 0 → 1B run.** No head of
the 270 crosses the sink threshold, at any of about 700 probes. The massive-activation
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
multimodal training while attention concentration never leaves zero (n = 1 seed; §5).**

## 3.3 No consistent-sign head-level relationship between concentration and value-norm

A tight coupling between concentration and value-drain should at minimum hold the sign of
their per-head correlation constant across training regimes. It does not. Table 3 reports
the Pearson r between attention→pos0 and value-norm ratio at the final checkpoint of each
arm, over the 90 (layer, KV-group) observations: under grouped-query attention a value
vector is shared by 3 query heads, so we average the attention of a group's query heads
rather than triplicate its value observation (§2). Appendix Fig. A2 plots the scatter.

**Table 3 — r(attn→pos0, v-ratio), final checkpoint. Descriptive only; we report no
p-values, for the reason given below.**

| baseline | g1gate | sigmoid | textinit | RF | pooled |
|---|---|---|---|---|---|
| +0.76 | +0.57 | −0.04 | −0.79 | +0.51 | −0.20 |

These correlations are **descriptive statistics, not inferential ones**, and we deliberately
report no significance tests: heads within a layer are not independent, and each arm rests
on a single seed at this checkpoint, so a p-value would mislead. Collapsing to the 90 KV
groups removes the pseudoreplication a per-query-head reading would introduce and leaves the
picture unchanged (Appendix D).

The pattern is about **sign**, which the pseudoreplication does not manufacture. Heads that
attend more to position 0 have *larger* value norms in the baseline regime and *smaller*
value norms under text initialization. The pooled correlation is weak only because arms of
opposite sign cancel, and it is not an uncorrelated cloud. We therefore claim no
consistent-sign relationship across arms. We do not claim the absence of a coupling law: a
fixed coupling would have to show one sign everywhere, and it does not.

## 3.4 Ordering: norm signatures lead, and concentration is late, mirrored, or absent

<figure id="fig2">
<img src="figures/fig4_birth_leadlag.svg" alt="Sink birth and lead-lag ordering">
<figcaption><b>Figure 2: Sink birth and ordering.</b> Seed 0 throughout. <em>Top:</em> a
time-to-event view. Each shaded track spans the tokens for which we observed that arm, from
60M to 1B. A filled marker shows the first probe at which a signature crossed its per-head
threshold: residual-norm ratio h&gt;2, value-drain v&lt;0.8, concentration
attn→pos0&gt;0.3. A hollow marker at the end of a track means that the signature never
crossed inside the observed run. <b>All crossing times are interval-censored</b> at the
100-step probe cadence: a reported birth step means "the first probe at which the threshold
had already been crossed". Two signatures that cross within one interval of each other
should not be read as ordered. <em>Bottom:</em> birth-maps, which show the step at which
each (layer, query-head) pair first crosses the concentration threshold. No baseline head
and no RF head ever crosses. 1% of g1gate heads cross, against 89% for sigmoid and 87% for
textinit.</figcaption>
</figure>

Dense probing lets us timestamp the arrival of each signature (Figure 2), and the ordering
stays consistent inside each arm family. In the softmax arms with randomly initialized
decoders, *baseline* and *g1gate* and *RF*, the norm signatures come early. The
residual-norm ratio crosses its per-head threshold within the first few to about 15M
tokens, and value-drain follows within about 50M tokens. Concentration never comes. No
baseline head and no RF head crosses the concentration threshold at any point in training,
and g1gate reaches 1% of heads, a single-layer blip late in training. *sigmoid* mirrors that
pattern: its concentration crosses early, at about 6M tokens, and 89% of its heads cross
eventually, while neither norm signature ever crosses. *textinit* inherits its norm
signatures rather than grows them. Value-drain and an elevated residual-norm ratio are both
present at 0 tokens, imported with the pretrained text-LM weights. Concentration is *not*
inherited the same way: Sink^0.3 is 0.000 at step 0 and crosses before 1M tokens. Even in
the arm that imports the most sink structure, the norm signatures precede concentration.

<figure id="fig3">
<img src="figures/fig6_sink_stripe.svg" alt="Query-by-key attention maps, top sink head per arm">
<figcaption><b>Figure 3: The sink stripe.</b> Query &times; key attention of the top sink
head of each arm, row-normalized, at seed 0; every panel is labelled with its seed, step and
token count. The top row is early in training (step 250, about 2.4M tokens). The bottom row
is the final checkpoint. The cyan line marks key = pos0 and the dotted line the
image-to-text boundary. The pos0 stripe is <em>absent</em> in baseline throughout, and
strong in textinit, which reaches attn→pos0 = 0.62 at its final checkpoint. The rightmost
panel shows the stripe <em>already present at step 0</em> in textinit, inherited from the
text language model. The <em>sigmoid</em> column uses head <b>L7H3</b> (0-indexed), the
arm's top sink head on the fixed probe batch at 0.87 row-normalized attention to pos0; its
heat maps are re-rendered from an auxiliary streaming batch, while every printed sigmoid
number comes from the fixed probe batch (Appendix D).</figcaption>
</figure>

Figure 3 shows the same result at the level of a single head, through the query × key
attention map of the top sink head of each arm. The vertical stripe at key = pos0 marks
every query attending to the first image token. That stripe is absent in *baseline* at both
checkpoints and strong in *textinit*. The rightmost panel carries the most weight. **The
stripe is already there at step 0 in textinit**, imported with the text-LM weights before
the model has seen one image.

The *sigmoid* arm needs one measurement note, because row-normalization changes the object
being measured and Gu et al. [6] state their result for *unnormalized* sigmoid attention.
The concentration this arm develops is a **relative** reallocation of a shrinking gate
budget onto position 0, not the growth of a large absolute mass there. Appendix D gives the
raw and row-normalized values behind that reading. We report the row-normalized view in the
tables so the arms stay comparable.

Attention-entropy collapse, which the text-LM literature uses as the usual correlate of
concentration, separates the same way. Only *sigmoid* and *textinit* collapse (Appendix
Fig. A3). Entropy collapse tracks the concentration axis, not the norm axes.

**The signatures also dissociate in position.** A per-position scan across all three
*textinit* seeds shows that the three signatures need not share a token. They coincide at
position 0 at seed 0; at seed 1 the attention maximum stays at position 0 while the residual
peak moves to pos1 and the value minimum to pos5; at seed 2 they separate furthest, with
attention at **pos1** and both norm extrema at **pos13** (Appendix C). The arms with
randomly initialized decoders behave differently. Position 0 remains the maximum-mass token
in *baseline*, *g1gate*, and *sigmoid* at every seed we scanned, and in *RF* the attention
argmax sits at pos1 with mass 0.100 against 0.083 at position 0, a diffuse profile rather
than a concentration sink.

We report this as a supporting observation, not a second headline. It carries one
consequence for measurement: because all three metrics anchor on position 0, the *textinit*
magnitudes at seeds 1 and 2 are read at a token that is no longer the peak and therefore
**understate** the arm's true peaks. That is a further reason to report *textinit* as a
range and a median (§3.1, §5). The anchoring holds for the arms with randomly initialized
decoders, which carry the central negative result.

Appendix E sets out, as untested hypotheses, how each lever might touch the sum-to-one
mechanism of Gu et al. [6] and thereby move a different axis. Nothing in this paper tests
them.
