# 3. Results

<figure id="fig1">
<img src="figures/fig2_phase_portrait.svg" alt="Decoupling phase portrait">
<figcaption><b>Figure 1: Decoupling phase-portrait.</b> Training trajectories of every arm
in (concentration &times; value-norm) space, at left, and in (concentration &times;
massive-activation) space, at right, on a log scale. Circles mark initialization. Diamonds
mark the final checkpoint. Each lever drives the signatures to a different corner.
<em>sigmoid</em> (red) reaches strong concentration, <em>amplifies</em> the value norms,
and shows no massive activation. <em>textinit</em> (orange) combines total concentration
with severe value-drain and an extreme massive activation.
<em>baseline</em>, <em>g1gate</em>, and <em>RF</em> (blue, green, violet) never leave the
no-concentration region, and their massive activation grows. If the three signatures were
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
270 (layer, head) pairs whose mean attention→pos0 exceeds 0.2.

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

| arm | concentration | value-norm | massive activation |
|---|---|---|---|
| baseline | absent (0.000) | mild drain (0.69–0.72) | moderate (1.7–2.2) |
| g1gate | suppressed (0.004–0.011) | **neutral** (0.81–0.85) | moderate (1.7–2.2) |
| sigmoid | strong (0.76–0.83) | **amplified** (1.48–1.60) | none (1.1–1.3) |
| textinit | total (0.56–0.85) | strong drain (0.38–0.63) | extreme (5.5–42.5) |

The corners separate pairwise on single axes. *g1gate* differs from *baseline* on the
value-norm axis alone. Concentration is absent or nearly absent in both arms, and massive
activation is moderate in both. The gate nonetheless neutralizes the mild value-drain that
the baseline develops. *sigmoid* and *textinit* both reach strong concentration, and they then
part on two axes. They differ in the *direction* of the value-norm move, amplified against
drained, and they differ in massive activation, none against extreme. One lever moves one
axis and leaves the other axes where a different lever put them. Figure 1 shows the four
trajectories fanning out from a common origin.

Reproducibility differs by arm. *g1gate* is the tightest arm. Its Sink^0.2 values are
0.004, 0.011, and 0.0037 across three seeds, which is at most 1% of heads. *baseline* and
*sigmoid* are tight in both of their seeds. The corner of *textinit* repeats across seeds. Its magnitudes do not. All three textinit seeds show high concentration, strong value-drain,
and an h-ratio above 5. Seed 0 nonetheless sits far above the other two seeds on every
signature at once. Sink reads 0.85 against 0.56–0.58, mean attn→pos0 reads 0.63 against
0.23, and h-ratio reads 42.5 against 5.5–12.2. The h-ratio has plateaued by 60–100M tokens in both lower
seeds. The spread is therefore genuine seed sensitivity, not an unconverged transient. We
report the massive activation of textinit as a **range (5.5–42.5×) with a median near 12×**
everywhere in this paper. We treat the corner as the reproducible claim, and no particular
magnitude.

## 3.2 Massive activation grows across 1B fresh tokens while concentration stays at zero

The four-arm comparison reuses a 146K-image pool at high epoch counts. A reader could
therefore read its "concentration never emerges" result as an overfitting artifact. The RF
arm removes that confound. It runs the identical *baseline* recipe on a fresh,
non-repeated FineVision stream to one billion tokens, which is 2.39 effective visual
epochs.

Concentration never arrives. **Sink^0.3 = 0.000 across the entire 0 → 1B run.** No head of
the 270 crosses the sink threshold, at any of about 700 probes. Massive activation, in
the same run, keeps growing far past warmup.

**Table 3 — RF (fresh data, seed 0), init → 1B tokens.**

| signature | @ init | @ 1B | net |
|---|---|---|---|
| Sink^0.3 (concentration) | 0.000 | 0.000 | flat zero, entire run |
| max attn→pos0 | 0.056 | 0.098 | ≈flat, far below threshold |
| h-ratio (massive activation) | 1.43 | **3.22** | **+130%**, still rising at 1B |
| v-ratio (value-norm) | 1.00 | 0.69 | net drain, non-monotone |

Warmup does not explain the rise in h-ratio. The h-ratio climbs monotonically from 2.40 at
about 57M tokens to 3.22 at 1B tokens, long after the 3% warmup window closes. In Figure 1,
at right, the violet trajectory climbs and never moves rightward. The v-ratio ends below
its value at initialization. About 75% of that drop happens in the first 57M tokens, and
the v-ratio then recovers part of it. We therefore report the v-ratio as supporting
context, not as ongoing emergence. **Massive activation grows across a full billion tokens
of multimodal training while attention concentration never leaves zero.** This run uses a
single seed (§5).

## 3.3 No universal head-level coupling between concentration and value-norm

Tight coupling between concentration and value-drain should at minimum hold the sign of
their per-head correlation constant across training regimes. It does not. Table 4 reports
the Pearson r between per-head attention→pos0 and value-norm ratio. We read it at the final
checkpoint of each arm, over n = 270 (layer, query-head) pairs per arm. Appendix Fig. A2
plots the scatter.

**Table 4 — per-head r(attn→pos0, v-ratio), final checkpoint.** ***p < 0.001; ns = not
significant. See the independence note below.

| baseline | g1gate | sigmoid | textinit | RF | pooled (n = 1350) |
|---|---|---|---|---|---|
| +0.67*** | +0.53*** | −0.03 ns | −0.76*** | +0.43*** | −0.20*** |

Two honesty notes apply to the n. First, the decoder uses grouped-query attention, where 9
query heads share 3 KV heads per layer (§2). The 270 pairs per arm therefore hold only 90
independent value-norm observations, because each KV group shares its value vector across 3
query heads. The attention side is genuinely per-query-head. Second, the significance stars
treat the pairs as independent, and heads within a layer and within a KV group are not
independent. Read the stars as descriptive, not as inferential. Neither caveat touches the
finding itself, which is about sign. Heads that attend more to position 0 have *larger*
value norms in the baseline regime and *smaller* value norms under text initialization. The
pooled correlation is weak only because arms of opposite sign cancel. A fixed-sign coupling
would show the same sign everywhere. Each lever instead sets its own local relationship
between the two axes.

## 3.4 Ordering: norm signatures lead, and concentration is late, mirrored, or absent

<figure id="fig2">
<img src="figures/fig4_birth_leadlag.svg" alt="Sink birth and lead-lag ordering">
<figcaption><b>Figure 2: Sink birth and ordering.</b> <em>Top:</em> a time-to-event view.
Each shaded track spans the tokens for which we observed that arm, from 60M to 1B. A filled
marker shows the first probe at which a signature crossed its per-head threshold: massive
activation h&gt;2, value-drain v&lt;0.8, concentration attn→pos0&gt;0.3. A hollow marker at
the end of a track means that the signature never crossed inside the observed run. In the
softmax-from-scratch arms, baseline and g1gate and RF, the norm signatures cross within
about 4&ndash;50M tokens. Concentration in those arms never crosses, through 174M tokens
for baseline and through 1B tokens for RF, or barely crosses, in g1gate. <em>sigmoid</em>
mirrors that pattern. Its concentration crosses at about 6M tokens, and neither norm
signature ever crosses. <em>textinit</em> inherits massive activation and value-drain at 0
tokens and crosses concentration before 1M tokens. <em>Bottom:</em> birth-maps, which show
the step at which each (layer, head) pair first crosses the concentration threshold. No
baseline head and no RF head ever crosses. 1% of g1gate heads cross, against 89% for
sigmoid and 87% for textinit.</figcaption>
</figure>

Dense probing lets us timestamp the arrival of each signature (Figure 2). The ordering
stays consistent inside each arm family. In the softmax-from-scratch arms, *baseline* and
*g1gate* and *RF*, the norm signatures come early. Massive activation crosses its per-head
threshold within the first few to about 15M tokens, and value-drain follows within about
50M tokens. Concentration never comes. No baseline head and no RF head crosses the
concentration threshold at any point in training. g1gate reaches 1% of heads, a
single-layer blip late in training. *sigmoid* mirrors that pattern. Its concentration
crosses early, at about 6M tokens, and 89% of its heads cross eventually, while neither
norm signature ever crosses. *textinit* inherits its signatures rather than grows them.
Value-drain is present at 0 tokens, imported with the pretrained text-LM weights, and
concentration crosses before 1M tokens.

<figure id="fig3">
<img src="figures/fig6_sink_stripe.svg" alt="Query-by-key attention maps, top sink head per arm">
<figcaption><b>Figure 3: The sink stripe.</b> Query &times; key attention of the top sink
head of each arm, row-normalized. The top row is early in training. The bottom row is the
final checkpoint. The cyan line marks key = pos0. The dotted line marks the
image-to-text boundary. The pos0 stripe is <em>absent</em> in baseline throughout, and
<em>total</em> in textinit, which reaches attn→pos0 = 0.62 at its final checkpoint. The
rightmost panel shows the stripe <em>already present at step 0</em> in textinit, inherited
from the text language model. We omit sigmoid. Its checkpoint dump selected heads by raw,
unnormalized gate score and missed the true top sink head of the arm, L7H3, which reaches
0.87 normalized. A panel built from the dumped heads would therefore understate the
concentration of sigmoid. Tables 1&ndash;2 and Figure 2 establish that concentration
instead.</figcaption>
</figure>

Figure 3 shows the same result at the level of a single head. It plots the query × key
attention map of the top sink head of each arm. The vertical stripe at key = pos0 marks
every query attending to the first image token. That stripe is absent in *baseline* at both
the early and the final checkpoint, and total in *textinit*. The rightmost panel carries
the most weight. **The stripe is already there at step 0 in textinit**, imported with the
text-LM weights before the model has seen one image. We omit *sigmoid* from this figure for
the measurement reason that the caption gives. Tables 1–2 and Figure 2 carry the
concentration result for that arm. Attention-entropy collapse, which the text-LM literature
uses as the usual correlate of concentration, separates the same way. Only *sigmoid* and
*textinit* collapse (Appendix Fig. A3). Entropy collapse tracks the concentration axis, not
the norm axes.

**A positional footnote.** In *textinit* the signatures partly decouple in *position* as
well as in magnitude. In some seeds the most-attended token, the residual-norm peak, and
the value-drain minimum sit on three different tokens. Attention sits at pos1, for example,
while the norm peak and the drain sit at pos13. We report this as a caveat on where to
anchor the magnitudes of textinit (§5), not as an independent finding.
