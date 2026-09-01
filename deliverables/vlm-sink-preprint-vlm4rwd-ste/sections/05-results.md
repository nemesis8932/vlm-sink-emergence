# 4. Results

<figure id="fig1">
<img src="figures/fig2_phase_portrait.svg" alt="Decoupling phase portrait">
<figcaption><b>Figure 1: Decoupling phase-portrait.</b> Every arm in
concentration-against-value-norm space (left) and against the residual-norm ratio (right,
log scale). Circles mark initialization, diamonds the final checkpoint (about 100M tokens,
60M for <em>textinit</em>, 1B for <em>RF</em>), seed 0 unless labelled. The horizontal axis
is the <em>maximum</em> attention→pos0 over heads, not Table 1's fraction above threshold,
so an arm can sit off zero here with Sink<sup>ε</sup> = 0.000. Paths smoothed, end markers
raw (Appendix F). Were the three signatures one phenomenon, these trajectories would move
along one direction.</figcaption>
</figure>

## 4.1 Four training levers reach four different signature corners

Table 1 collapses the seeds into per-arm ranges, each arm at its final matched checkpoint,
about 100M tokens or 60M for *textinit*, whose signatures have plateaued by then (per-seed
table in Appendix D). No two arms share a signature triple.

**Table 1: the four corners (n = 2–3 seeds per arm).** Concentration is Sink^0.2, the
fraction of the 270 (layer, query-head) pairs whose mean attention→pos0 exceeds 0.2. The
v-ratio rests on 90 (layer, KV-group) value projections and the h-ratio on 30 layers (§3).

| arm | concentration | value-norm | massive-activation proxy |
|---|---|---|---|
| baseline | absent (0.000) | mild drain (0.69–0.72) | moderate (1.7–2.2) |
| g1gate | near-absent (0.004–0.011) | **milder drain** (0.81–0.85) | moderate (1.7–2.2) |
| sigmoid | strong (0.76–0.83) | **amplified** (1.48–1.60) | no strong asymmetry (1.1–1.3) |
| textinit | strong (0.56–0.85) | strong drain (0.38–0.63) | extreme (5.5–42.5) |

The value-norm axis alone takes three directions, drained hard, drained mildly and
amplified, and the corners separate pairwise on single axes. *g1gate* differs from
*baseline* on the value-norm axis alone. Concentration is absent or near-absent and the
residual-norm ratio moderate in both, while the gate makes the baseline's mild drain milder
still, a 15–19% drain rather than none. *sigmoid* and *textinit* both reach strong
concentration and then part on the direction of the value-norm move and on the residual-norm
ratio. The *sigmoid* corner is relative. Its concentration is row-normalized over a
shrinking raw gate budget, with top-head raw pos0 mass 0.065 at the final checkpoint
(Appendix F). The arms are not a factorial design, so these are intervention-associated
profiles, not isolated causal effects of single levers. The trajectories in Figure 1 separate
early and do not share one origin. *textinit* starts from a pretrained text decoder that
already carries a subthreshold first-position bias, with Sink^0.3 still 0.000 at step 0.

Reproducibility differs by arm. *g1gate* is tightest, Sink^0.2 of 0.004, 0.011 and 0.0037
across three seeds. The corner of *textinit* repeats across seeds and its magnitudes do not.
Seed 0 sits far above the other two on every signature at once, h-ratio 42.5 against
5.5–12.2, partly because at seeds 1 and 2 its peak signatures move off position 0, where our
metrics anchor (Appendix D, I). We therefore report textinit's massive-activation proxy as a
range, 5.5–42.5× with a median near 12×, and treat the corner as the reproducible claim
rather than any magnitude.

The corners also separate in time (Fig. A4, seed 0, Appendix I). The random-decoder softmax
arms *baseline*, *g1gate* and *RF* cross the norm thresholds without ever crossing
concentration. *sigmoid* is the mirror image, crossing concentration without either norm
threshold. *textinit* starts above both norm thresholds and crosses concentration later.
Where a run crosses both kinds, the norm signatures come first, and under text
initialization the three need not even settle on the same token. Appendix I gives the
timings, the positional scan and the entropy-collapse correlate, and Appendix G sets out, as
untested hypotheses, how each lever might move a different axis.

Figure 2 shows the separation inside a single head. Its rightmost panel carries the most
weight. The textinit stripe is already there at step 0, imported with the text-LM weights
before the model has seen one image.

<figure id="fig2">
<img src="figures/fig6_sink_stripe.svg" alt="Query-by-key attention maps, top sink head per arm">
<figcaption><b>Figure 2: The sink stripe.</b> Query &times; key attention of each arm's top
sink head, row-normalized, seed 0. Top row is early training (step 250, about 2.4M tokens),
bottom row the final checkpoint. The cyan line marks key = pos0, the dotted line the
image-to-text boundary. The stripe is absent in baseline throughout, strong in textinit
(attn→pos0 = 0.62 at its final checkpoint), and already present at step 0 in textinit
(rightmost panel), inherited from the text LM. The sigmoid heat maps come from an auxiliary
streaming batch, while every printed sigmoid number comes from the fixed probe batch
(Appendix F).</figcaption>
</figure>

## 4.2 The proxy grows across 1B low-repetition tokens with concentration at zero

The four-arm comparison reuses a 146K-image pool, so a reader could read its "concentration
never emerges" result as an overfitting artifact. The RF arm answers that. It runs the
identical *baseline* recipe on a fresh FineVision stream to one billion tokens at 2.39
effective visual epochs, a low-repetition run rather than a repetition-free one, with a
held-out loss that ends at 0.638 on a negative fitted slope over the second half, while
individual evaluations fluctuate (§3, §5).

Concentration never arrives. **Sink^0.3 = 0.000 across the entire 0 → 1B run**, and no head
of the 270 crosses the threshold at any of about 700 probes.

**Table 2: RF (fresh stream, n = 1 seed), init → 1B tokens.**

| signature | @ init | @ 1B | net |
|---|---|---|---|
| Sink^0.3 (concentration) | 0.000 | 0.000 | flat zero, entire run |
| max attn→pos0 | 0.056 | 0.098 | ≈flat, far below threshold |
| h-ratio (massive-activation proxy) | 1.43 | **3.22** | **≈2.3×**, positive long-horizon trend |
| v-ratio (value-norm) | 1.00 | 0.69 | net drain, non-monotone |

Warmup does not explain the rise. The h-ratio climbs from 2.40 at about 57M tokens to 3.22
at 1B, long after the 3% warmup window closes. The climb is not monotonic at probe
resolution, so we call it a positive long-horizon trend. In Figure 1, right, the violet
trajectory climbs and never moves rightward. The v-ratio ends below its initialization
value, but about 75% of that drop happens in the first 57M tokens and part then recovers, so
we report it as supporting context rather than ongoing emergence.

## 4.3 No consistent-sign head-level concentration–value-norm relationship

A tight coupling between concentration and value-drain should at minimum hold the sign of
their per-head correlation constant across regimes. It does not. Table 3 gives the Pearson r
between attention→pos0 and value-norm ratio at each arm's final checkpoint, seed 0, over the
90 (layer, KV-group) observations. We average a group's query-head attention rather than
triplicate its value observation (§3, scatter in Fig. A2, Appendix C).

**Table 3: r(attn→pos0, v-ratio), final checkpoint, seed 0.**

| baseline | g1gate | sigmoid | textinit | RF | pooled |
|---|---|---|---|---|---|
| +0.76 | +0.57 | −0.03 | −0.79 | +0.51 | −0.20 |

These correlations are descriptive, not inferential. Heads within a layer are not
independent, and each arm rests on a single seed here, so a p-value would mislead and we
report none. Collapsing to the 90 KV groups removes the pseudoreplication a per-query-head
reading would introduce, and the picture does not change (Appendix F).

The pattern is about sign, which pseudoreplication does not manufacture. Heads attending
more to position 0 have larger value norms in the baseline regime and smaller ones under
text initialization, and the pooled correlation is weak only because arms of opposite sign
cancel. We claim that no consistent-sign relationship holds across arms, not that no
coupling law exists. A fixed coupling would show one sign everywhere, and it does not.
