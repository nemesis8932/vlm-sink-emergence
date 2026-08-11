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
metrics anchor (Appendix B, Appendix H). We therefore report the massive-activation proxy of
textinit as a **range (5.5–42.5×) with a median near 12×** throughout, and treat the corner
as the reproducible claim rather than any magnitude.

**The corners also form in a consistent order.** Probes every 100 steps timestamp each
signature's arrival (Figure 2). In the softmax arms with randomly initialized decoders,
*baseline* and *g1gate* and *RF*, the norm signatures cross first and concentration never
crosses at all. *sigmoid* mirrors that, crossing on concentration alone. *textinit* imports
both norm signatures at 0 tokens and grows concentration after. In every arm the norm
signatures precede concentration, and under text initialization the three need not even
settle on the same token. Appendix H gives the timings, the positional scan and the
entropy-collapse correlate, which follows the concentration axis alone.

<figure id="fig2">
<img src="figures/fig4_leadlag_top.svg" alt="Sink lead-lag ordering">
<figcaption><b>Figure 2: When each signature first crosses threshold.</b> Seed 0 throughout.
Time-to-event tracks spanning the tokens over which we observed each arm (60M to 1B). A
filled marker is the first probe at which a signature crossed its per-head threshold
(h&gt;2, v&lt;0.8, attn→pos0&gt;0.3). A hollow marker at a track's end means it never
crossed. <b>Crossing times are interval-censored</b> at the 100-step probe cadence, so two
signatures crossing within one interval should not be read as ordered. Appendix H, Fig. A4 maps the
crossings head by head.</figcaption>
</figure>

Figure 3 shows the separation inside a single head. The vertical stripe at key = pos0 is
*absent* in *baseline* at both checkpoints and strong in *textinit*, where the rightmost
panel carries the most weight: **the stripe is already there at step 0**, imported with the
text-LM weights before the model has seen one image.

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

## 3.2 The massive-activation proxy grows across 1B low-repetition tokens while concentration stays at zero

The four-arm comparison reuses a 146K-image pool, so a reader could read its "concentration
never emerges" result as an overfitting artifact. The RF arm addresses that: the identical
*baseline* recipe on a fresh FineVision stream to one billion tokens, at **2.39 effective
visual epochs**, a **low-repetition** run rather than a repetition-free one, whose held-out
loss falls throughout and never turns upward (§2, §5).

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
