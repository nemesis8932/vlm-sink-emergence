# 3. Results

<figure id="fig1">
<img src="figures/fig2_phase_portrait.svg" alt="Decoupling phase portrait">
<figcaption><b>Figure 1: Decoupling phase-portrait.</b> Training trajectories of every arm
in concentration-against-value-norm space, at left, and concentration against the
residual-norm ratio, at right, on a log scale. <b>Note the metric difference
from the tables:</b> the horizontal axis here is the <em>maximum</em> attention→pos0 over
heads, a continuous quantity that keeps a trajectory visible, whereas Tables 1&ndash;2
report Sink<sup>ε</sup><sub>1</sub>, the <em>fraction of query heads above a threshold</em>.
An arm can therefore show a non-zero position on this axis while its Sink<sup>ε</sup>
remains 0.000. The value-norm ratio is per KV head and the h-ratio is per layer (&sect;2).
Paths are smoothed with a moving average, 5 points on the horizontal axis and 9 on the
vertical; the faint dots behind each path are the unsmoothed per-probe values, and the
init and final markers use raw values. Circles mark initialization. Diamonds mark the
final checkpoint, at about 100M tokens for <em>baseline</em>, <em>g1gate</em> and
<em>sigmoid</em>, 60M for <em>textinit</em>, and 1B for <em>RF</em>; seed 0 unless a seed is
labelled. Each lever drives the signatures to a different corner. <em>sigmoid</em> (red)
reaches strong concentration, <em>amplifies</em> the value norms, and shows no strong
residual-norm asymmetry. <em>textinit</em> (orange) combines strong concentration with
severe value-drain and an extreme residual-norm ratio.
<em>baseline</em>, <em>g1gate</em>, and <em>RF</em> (blue, green, violet) never leave the
no-concentration region, and their residual-norm ratio grows. If the three signatures were
one phenomenon, these trajectories would move together along one direction. Instead every
combination of directions occurs.</figcaption>
</figure>

## 3.1 Four training levers reach four different signature corners

Table 1 reports the three signatures for all four arms and all seeds. *baseline* and
*sigmoid* have two seeds each. *g1gate* and *textinit* have three. Each row is read at its
final matched checkpoint: about 100M tokens for baseline, g1gate, and sigmoid, and 60M
tokens for *textinit*. All three signatures of *textinit* have plateaued at 60M tokens. Its
h-ratio changes by less than 10% from 60M to 100M in the seeds that continued. We trust the
seed-0 values from a checksummed archive rather than re-derive them first-hand (§5).

**Table 1 — signature triple by arm, all seeds.** Sink^0.2 is the fraction of the model's
270 (layer, query-head) pairs whose mean attention→pos0 exceeds 0.2. The v-ratio aggregates
90 (layer, KV-head) value projections and the h-ratio aggregates 30 layers (§2).

| arm | seed | tokens | Sink^0.2 | mean attn→pos0 | v-ratio | h-ratio |
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

Table 2 collapses the seeds into per-arm ranges. **No two arms share a signature triple.**
The value-norm axis alone takes three qualitatively different directions. The lever drains
it below 1, leaves it near 1, or amplifies it above 1.

**Table 2 — the four corners.**

| arm | concentration | value-norm | massive-activation proxy |
|---|---|---|---|
| baseline | absent (0.000) | mild drain (0.69–0.72) | moderate (1.7–2.2) |
| g1gate | near-absent (0.004–0.011) | **milder drain** (0.81–0.85) | moderate (1.7–2.2) |
| sigmoid | strong (0.76–0.83) | **amplified** (1.48–1.60) | no strong asymmetry (1.1–1.3) |
| textinit | strong (0.56–0.85) | strong drain (0.38–0.63) | extreme (5.5–42.5) |

The corners separate pairwise on single axes. *g1gate* differs from *baseline* on the
value-norm axis alone. Concentration is absent or near-absent in both arms (note that
*g1gate* sits *above* the exact 0.000 of baseline, not below it), and the residual-norm
ratio is moderate in both. The gate nonetheless makes the mild value-drain of the baseline
milder still, from 0.69–0.72 to 0.81–0.85, which is a 15–19% drain rather than none.
*sigmoid* and *textinit* both reach strong concentration, and they then part on two axes.
They differ in the *direction* of the value-norm move, amplified against drained, and they
differ in the residual-norm ratio, absent against extreme. Each arm shows a distinct
intervention-associated profile. Because the four arms are not a factorial design, we do
not attribute these profiles to isolated causal effects of single levers. Figure 1 shows
the four trajectories fanning out from a common origin.

Reproducibility differs by arm. *g1gate* is the tightest arm. Its Sink^0.2 values are
0.004, 0.011, and 0.0037 across three seeds, which is at most 1% of heads. *baseline* and
*sigmoid* are tight in both of their seeds. The corner of *textinit* repeats across seeds. Its magnitudes do not. All three textinit seeds show high concentration, strong value-drain,
and an h-ratio above 5. Seed 0 nonetheless sits far above the other two seeds on every
signature at once. Sink reads 0.85 against 0.56–0.58, mean attn→pos0 reads 0.63 against
0.23, and h-ratio reads 42.5 against 5.5–12.2. The h-ratio has plateaued by 60–100M tokens in both lower
seeds. The spread is therefore genuine seed sensitivity, not an unconverged transient. Part
of it is also positional: at seeds 1 and 2 the arm's peak signatures move off position 0,
where our metrics are anchored (§3.4). We report the massive-activation proxy of textinit
as a **range (5.5–42.5×) with a median near 12×** everywhere in this paper. We treat the
corner as the reproducible claim, and no particular magnitude.

## 3.2 The massive-activation proxy grows across 1B low-repetition tokens while concentration stays at zero

The four-arm comparison reuses a 146K-image pool. A reader could therefore read its
"concentration never emerges" result as an overfitting artifact. The RF arm addresses that.
It runs the identical *baseline* recipe on a fresh FineVision stream to one billion tokens,
at **2.39 effective visual epochs**. Examples still repeat, about 2.4 times on average, so
this is a low-repetition run rather than a repetition-free one. What changes is the
overfitting signature: held-out validation tracks train loss, and the large `val_seen` /
`val_unseen` gap of the repeated arms disappears (§2).

Concentration never arrives. **Sink^0.3 = 0.000 across the entire 0 → 1B run.** No head of
the 270 crosses the sink threshold, at any of about 700 probes. The massive-activation
proxy, in the same run, keeps growing far past warmup.

**Table 3 — RF (fresh stream, 2.39 visual epochs, seed 0), init → 1B tokens.**

| signature | @ init | @ 1B | net |
|---|---|---|---|
| Sink^0.3 (concentration) | 0.000 | 0.000 | flat zero, entire run |
| max attn→pos0 | 0.056 | 0.098 | ≈flat, far below threshold |
| h-ratio (massive-activation proxy) | 1.43 | **3.22** | **≈2.3×**, positive long-horizon trend |
| v-ratio (value-norm) | 1.00 | 0.69 | net drain, non-monotone |

Warmup does not explain the rise in h-ratio. The h-ratio climbs from 2.40 at about 57M
tokens to 3.22 at 1B tokens, long after the 3% warmup window closes. The rise is not
monotone at probe resolution; we report it as a positive long-horizon trend. In Figure 1,
at right, the violet trajectory climbs and never moves rightward. The v-ratio ends below
its value at initialization. About 75% of that drop happens in the first 57M tokens, and
the v-ratio then recovers part of it. We therefore report the v-ratio as supporting
context, not as ongoing emergence. **The massive-activation proxy grows about 2.3× across a
full billion tokens of multimodal training while attention concentration never leaves
zero.** This run uses a single seed (§5).

## 3.3 No consistent-sign head-level relationship between concentration and value-norm

A tight coupling between concentration and value-drain should at minimum hold the sign of
their per-head correlation constant across training regimes. It does not. Table 4 reports
the Pearson r between per-head attention→pos0 and value-norm ratio. We read it at the final
checkpoint of each arm, over the 270 (layer, query-head) pairs per arm. Appendix Fig. A2
plots the scatter.

**Table 4 — per-head r(attn→pos0, v-ratio), final checkpoint. Descriptive only; we report
no p-values, for the reason given below.**

| baseline | g1gate | sigmoid | textinit | RF | pooled |
|---|---|---|---|---|---|
| +0.67 | +0.53 | −0.03 | −0.76 | +0.43 | −0.20 |

These correlations are **descriptive statistics, not inferential ones**, and we deliberately
report no significance tests. The observations are not independent in two ways. The decoder
uses grouped-query attention, where 9 query heads share 3 KV heads per layer (§2), so the
270 pairs per arm hold only 90 independent value-norm observations: each KV group repeats
its value vector across 3 query heads. Heads within a layer are not independent either. Any
p-value computed over 270 nominally independent pairs would therefore be inflated by
pseudoreplication, and each arm additionally rests on a single seed at this checkpoint. The
attention side is genuinely per-query-head.

The pattern we report is about **sign**, which the pseudoreplication does not manufacture.
Heads that attend more to position 0 have *larger* value norms in the baseline regime and
*smaller* value norms under text initialization. The pooled correlation is weak only
because arms of opposite sign cancel; it is not an uncorrelated cloud. We therefore claim
no consistent-sign relationship across arms, and we do not claim the absence of a coupling
law: a fixed coupling would have to show one sign everywhere, and it does not.

## 3.4 Ordering: norm signatures lead, and concentration is late, mirrored, or absent

<figure id="fig2">
<img src="figures/fig4_birth_leadlag.svg" alt="Sink birth and lead-lag ordering">
<figcaption><b>Figure 2: Sink birth and ordering.</b> <em>Top:</em> a time-to-event view.
Each shaded track spans the tokens for which we observed that arm, from 60M to 1B. A filled
marker shows the first probe at which a signature crossed its per-head threshold: massive
activation h&gt;2, value-drain v&lt;0.8, concentration attn→pos0&gt;0.3. A hollow marker at
the end of a track means that the signature never crossed inside the observed run. In the
softmax arms with random decoders, baseline and g1gate and RF, the norm signatures cross within
about 4&ndash;50M tokens. Concentration in those arms never crosses, through 174M tokens
for baseline and through 1B tokens for RF, or barely crosses, in g1gate. <em>sigmoid</em>
mirrors that pattern. Its concentration crosses at about 6M tokens, and neither norm
signature ever crosses. <em>textinit</em> inherits its norm signatures at 0
tokens and crosses concentration before 1M tokens. <em>Bottom:</em> birth-maps, which show
the step at which each (layer, query-head) pair first crosses the concentration threshold.
No baseline head and no RF head ever crosses. 1% of g1gate heads cross, against 89% for
sigmoid and 87% for textinit. <b>All crossing times are interval-censored</b> at the probe
cadence: a signature is observed only every 100 optimizer steps, so a reported birth step
means "the first probe at which the threshold had already been crossed", and the true
crossing lies somewhere inside the preceding interval. Two signatures that cross within one
interval of each other should not be read as ordered. Seed 0 throughout.</figcaption>
</figure>

Dense probing lets us timestamp the arrival of each signature (Figure 2). The ordering
stays consistent inside each arm family. In the softmax arms with random decoders, *baseline* and
*g1gate* and *RF*, the norm signatures come early. Massive activation crosses its per-head
threshold within the first few to about 15M tokens, and value-drain follows within about
50M tokens. Concentration never comes. No baseline head and no RF head crosses the
concentration threshold at any point in training. g1gate reaches 1% of heads, a
single-layer blip late in training. *sigmoid* mirrors that pattern. Its concentration
crosses early, at about 6M tokens, and 89% of its heads cross eventually, while neither
norm signature ever crosses. *textinit* inherits its norm signatures rather than grows
them. Value-drain and an elevated residual-norm ratio are both present at 0 tokens,
imported with the pretrained text-LM weights. Concentration is *not* inherited in the same
way: Sink^0.3 is 0.000 at step 0 and crosses before 1M tokens. Even in the arm that imports
the most sink structure, the norm signatures precede concentration.

<figure id="fig3">
<img src="figures/fig6_sink_stripe.svg" alt="Query-by-key attention maps, top sink head per arm">
<figcaption><b>Figure 3: The sink stripe.</b> Query &times; key attention of the top sink
head of each arm, row-normalized, at seed 0; every panel is labelled with its seed, step
and token count. The top row is early in training (step 250, about 2.4M tokens). The bottom
row is the final checkpoint. The cyan line marks key = pos0. The dotted line marks the
image-to-text boundary. The pos0 stripe is <em>absent</em> in baseline throughout, and
strong in textinit, which reaches attn→pos0 = 0.62 at its final checkpoint. The
<em>sigmoid</em> column uses head <b>L7H3</b> (0-indexed), the arm's true top sink head at
0.87 row-normalized attention to pos0; an earlier dump selected heads by raw, unnormalized
gate score and picked different heads, which understated the arm. The rightmost panel shows
the stripe <em>already present at step 0</em> in textinit, inherited from the text language
model. The pos0 share printed on each panel is computed over valid query positions, the same
convention as the tables.</figcaption>
</figure>

Figure 3 shows the same result at the level of a single head. It plots the query × key
attention map of the top sink head of each arm. The vertical stripe at key = pos0 marks
every query attending to the first image token. That stripe is absent in *baseline* at both
the early and the final checkpoint, and strong in *textinit*. The rightmost panel carries
the most weight. **The stripe is already there at step 0 in textinit**, imported with the
text-LM weights before the model has seen one image.

**Raw against row-normalized sigmoid.** The *sigmoid* panel needs one measurement note,
because row-normalization changes the object being measured and Gu et al. [1] state their
result for *unnormalized* sigmoid attention. Head L7H3 of the sigmoid arm sends 0.873 of its
row-normalized attention to position 0 at the final checkpoint, which is the arm maximum.
Its raw gate mass to position 0 is 0.052, and the raw mass summed over all keys in that row
is 0.083: the row does not sum to one, and pos0 takes about 62% of what little
gate mass the head opens at all. Early in training the same head shows the opposite picture:
raw pos0 mass 0.389 against a raw row sum of 13.5, so only about 3% of a very large gate
budget. The concentration this arm develops is therefore a **relative** reallocation of a
shrinking gate budget onto position 0, not the growth of a large absolute mass there. Both
readings appear in `analysis/sigmoid_raw_vs_norm.json`. We report the row-normalized view in
the tables so the arms stay comparable, and we flag the raw view here because it is the
quantity Gu's claim concerns.

Attention-entropy collapse, which the text-LM literature
uses as the usual correlate of concentration, separates the same way. Only *sigmoid* and
*textinit* collapse (Appendix Fig. A3). Entropy collapse tracks the concentration axis, not
the norm axes.

**The signatures also dissociate in position.** A per-position scan of attention mass,
residual norms, and value norms across all three *textinit* seeds shows that the three
signatures need not share a token. At seed 0 they coincide: attention mass, the
residual-norm peak, and the value-norm minimum all sit at position 0. At seed 1 the
attention maximum stays at position 0 while the residual peak moves to pos1 and the value
minimum to pos5. At seed 2 they separate furthest: the attention maximum sits at **pos1**
while the residual peak and the value trough both sit at **pos13**. The softmax arms with
randomly initialized decoders behave differently: position 0 remains the maximum-mass token
in *baseline*, *g1gate*, and *sigmoid* at every seed we scanned (Appendix C). In *RF* the
attention argmax sits at pos1, but the mass there is 0.100 against 0.083 at position 0,
which is a diffuse profile in the baseline range rather than a concentration sink.

We report this as a supporting observation, not as a second headline. It carries one
consequence for measurement: because all three of our metrics anchor on position 0 by
construction, the *textinit* magnitudes at seeds 1 and 2 are read at a token that is no
longer the peak, so those pos0-anchored values **understate** the arm's true peak
magnitudes. That is a further reason to report *textinit* as a range and a median (§3.1,
§5). The pos0 anchoring holds for the arms with randomly initialized decoders, which carry
the central negative result.

## 3.5 Interpretation (hypotheses)

This subsection is **speculative**. Nothing in it is tested by our experiments, and none of
it is a claim of this paper. We offer it because a reader is entitled to ask *why* the
signatures come apart, and because these hypotheses are cheap to state and testable later.

Gu et al. [1] account for the sink as a key bias: softmax must distribute a full unit of
attention mass per row, and a head with nothing informative to retrieve parks the surplus
on a token whose value contributes little. If that account is right, each of our levers
touches a different part of it, which would explain why each moves a different axis.

- *g1gate.* An output gate can suppress whatever the attended position injects into the
  residual stream. A model with that gate may be able to *afford* concentration, because
  concentration no longer forces a matching change in the value path. That would be
  consistent with a gate whose measured effect here is on the value-norm axis rather than
  the concentration axis. Fesser et al. [25] make a compatible argument from another
  direction: they distinguish a sink that acts as an "adaptive nop", recognizable by a
  negligible value norm, from a sink that broadcasts global information, and they note that
  gating implicitly assumes the nop mechanism. If that is right, a gate should act on the
  value-norm axis first, which is where our gated arm differs from baseline.
- *sigmoid.* Removing the softmax removes the sum-to-one constraint itself. A head with
  nothing to retrieve can simply attend weakly to everything, with no surplus to park. On
  this reading the value amplification we observe is what the value path does when it is
  no longer compensating for a forced allocation.
- *textinit.* A pretrained text decoder arrives with sink machinery already built. Our
  observation that its norm signatures are present at step 0 while concentration crosses
  later would then reflect import of structure followed by re-targeting onto a visual
  prefix, rather than formation from scratch.

Each of these is a hypothesis about mechanism. Testing them needs interventions we did not
run: gate-scale sweeps that separate gating from the half-scale confound of §2, sigmoid
runs at matched effective attention temperature, and text-initialized runs with the sink
machinery ablated before alignment.
