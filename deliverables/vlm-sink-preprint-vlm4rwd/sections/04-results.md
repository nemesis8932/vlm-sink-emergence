# 3. Results

<figure id="fig1">
<img src="figures/fig2_phase_portrait.svg" alt="Decoupling phase portrait">
<figcaption><b>Figure 1: Decoupling phase-portrait.</b> Training trajectories of every arm
in (concentration × value-norm) space (left) and (concentration × massive-activation)
space (right, log scale); circles mark initialization, diamonds the final checkpoint. Each
lever drives the signatures into a distinct corner: <em>sigmoid</em> (red) reaches strong
concentration with value norms <em>amplified</em> and no massive activation;
<em>textinit</em> (orange) couples total concentration with severe value-drain and an
extreme massive activation; <em>baseline</em>/<em>g1gate</em>/<em>RF</em>
(blue/green/violet) never leave the no-concentration region while their massive activation
grows. Were the three signatures one phenomenon, these trajectories would co-move along a
single direction. Every combination of directions occurs instead.</figcaption>
</figure>

## 3.1 Four training levers land in four distinct signature corners

Table 1 reports the three signatures for all four arms and all seeds (baseline/sigmoid
n = 2; g1gate/textinit n = 3), each read at its final matched checkpoint: ~100M tokens for
baseline, g1gate, and sigmoid, and 60M for *textinit*, where all three of its signatures
have plateaued; its h-ratio changes by less than 10% from 60M to 100M in the seeds that
continued. Seed-0 values are trusted from a checksummed archive
rather than re-derived first-hand (§5).

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

Collapsing seeds to per-arm ranges (Table 2), **no two arms share a signature triple**.
The value-norm axis alone takes three qualitatively different directions: drained,
unchanged, amplified.

**Table 2 — the four corners.**

| arm | concentration | value-norm | massive activation |
|---|---|---|---|
| baseline | absent (0.000) | mild drain (0.69–0.72) | moderate (1.7–2.2) |
| g1gate | suppressed (0.004–0.011) | **neutral** (0.81–0.85) | moderate (1.7–2.2) |
| sigmoid | strong (0.76–0.83) | **amplified** (1.48–1.60) | none (1.1–1.3) |
| textinit | total (0.56–0.85) | strong drain (0.38–0.63) | extreme (5.5–42.5) |

The corners dissociate pairwise on single axes. *g1gate* separates from *baseline* purely
on value-norm: concentration is (near-)absent and massive activation moderate in both, but
the gate neutralizes the mild value-drain the baseline develops. *sigmoid* and *textinit*
both reach strong concentration yet part ways on the value-norm *direction* (amplified vs.
drained) and on massive activation (none vs. extreme). One lever moves one axis and leaves
the others where a different lever put them; Figure 1 shows the four trajectories fanning
out of a common origin.

Reproducibility differs by arm. *g1gate* is tightest (Sink^0.2 = 0.004/0.011/0.0037
across three seeds — at most 1% of heads); *baseline* and *sigmoid* (n = 2) are tight in
both seeds. *textinit*'s corner is robust but its magnitudes are not: all three seeds show
high concentration, strong value-drain, and h-ratio > 5, but seed 0 sits far above the
other two on every signature at once (Sink 0.85 vs. 0.56–0.58; mean attn→pos0 0.63 vs.
0.23; h-ratio 42.5 vs. 5.5–12.2). The h-ratio has plateaued by 60–100M tokens in both
lower seeds, so the spread is genuine seed sensitivity rather than an unconverged
transient. We therefore report textinit's massive activation as a
**range (5.5–42.5×), median ≈ 12×**, everywhere in this paper, and treat the corner as the
reproducible claim rather than any particular magnitude.

## 3.2 Massive activation grows for 1B fresh tokens while concentration stays at zero

The four-arm comparison reuses a 146K-image pool at high epoch counts, so its
"concentration never emerges" reading could conceivably be an overfitting artifact. The RF
run removes that confound: the identical *baseline* recipe on a fresh, non-repeated
FineVision stream to one billion tokens (2.39 effective visual epochs).

Concentration never arrives. **Sink^0.3 = 0.000 across the entire 0 → 1B run**; not one of
the 270 heads crosses the sink threshold at any of ~700 probes. Massive activation,
meanwhile, keeps growing far past warmup:

**Table 3 — RF (fresh data, seed 0), init → 1B tokens.**

| signature | @ init | @ 1B | net |
|---|---|---|---|
| Sink^0.3 (concentration) | 0.000 | 0.000 | flat zero, entire run |
| max attn→pos0 | 0.056 | 0.098 | ≈flat, far below threshold |
| h-ratio (massive activation) | 1.43 | **3.22** | **+130%**, still rising at 1B |
| v-ratio (value-norm) | 1.00 | 0.69 | net drain, non-monotone |

Warmup does not explain the h-ratio rise: it continues monotonically from 2.40 at ~57M
tokens to 3.22 at 1B, long after the 3% warmup window closes. In Figure 1 (right), the
violet trajectory climbs while never moving rightward. The v-ratio ends below its
initialization value, but ~75% of that drop happens in the first 57M tokens and partially
recovers afterward, so we report it as supporting context rather than ongoing emergence.
**Massive activation grows for a full billion tokens of multimodal training while
attention concentration never leaves zero** (a single seed; §5).

## 3.3 No universal head-level coupling between concentration and value-norm

If concentration and value-drain were tightly coupled, their per-head correlation should
at minimum carry a consistent sign across training regimes. It does not. Pearson r between
per-head attention→pos0 and value-norm ratio at each arm's final checkpoint (n = 270
(layer, query-head) pairs/arm; scatter in Appendix Fig. A2):

**Table 4 — per-head r(attn→pos0, v-ratio), final checkpoint.** ***p < 0.001; ns = not
significant. See the independence note below.

| baseline | g1gate | sigmoid | textinit | RF | pooled (n = 1350) |
|---|---|---|---|---|---|
| +0.67*** | +0.53*** | −0.03 ns | −0.76*** | +0.43*** | −0.20*** |

Two honesty notes on the n. The decoder uses grouped-query attention (9 query heads share
3 KV heads per layer; §2), so the 270 pairs per arm contain only 90 independent value-norm
observations — each KV group's value vector is shared by 3 query heads, while the
attention side is genuinely per-query-head. And the significance stars treat pairs as
independent, which heads within a layer and within a KV group are not; read the stars as
descriptive, not inferential. Neither caveat touches the finding itself, which is about
sign: heads that attend more to pos0 have *larger* value norms in the baseline regime and
*smaller* ones under text initialization, and the pooled correlation is weak only because
opposite-signed arms cancel. A fixed-sign coupling would show the same sign everywhere.
Each lever instead sets its own local relationship between the two axes.

## 3.4 Ordering: norm signatures lead, concentration is late, mirror-imaged, or never

<figure id="fig2">
<img src="figures/fig4_birth_leadlag.svg" alt="Sink birth and lead-lag ordering">
<figcaption><b>Figure 2: Sink birth &amp; ordering.</b> <em>Top:</em> time-to-event view.
Each shaded track spans the tokens an arm was actually observed for (60M–1B); a filled
marker is the first probe at which a signature crossed its per-head threshold (massive-act
h&gt;2, value-drain v&lt;0.8, concentration attn→pos0&gt;0.3); a hollow marker at a
track's end means that signature never crossed within the observed run. In the
softmax-from-scratch arms (baseline, g1gate, RF) the norm signatures cross within
~4–50M tokens while concentration never crosses (baseline through 174M, RF through 1B) or
barely does (g1gate). <em>sigmoid</em> is the mirror image: concentration at ~6M tokens,
both norm signatures never. <em>textinit</em> inherits massive-act and value-drain at 0
tokens and crosses concentration before 1M. <em>Bottom:</em> birth-maps, the step at which
each (layer, head) first crosses the concentration threshold. 0% of baseline and RF heads
ever cross; 1% of g1gate heads, vs. 89% (sigmoid) and 87% (textinit).</figcaption>
</figure>

Dense probing lets us timestamp each signature's arrival (Figure 2), and the ordering is
consistent within arm families. In the softmax-from-scratch arms (*baseline*, *g1gate*,
*RF*), the norm signatures come early: massive activation crosses its per-head threshold
within the first few to ~15M tokens, and value-drain follows within ~50M. Concentration
never comes. 0% of baseline and RF heads cross the concentration threshold at any point in
training; g1gate reaches 1% of heads, a single-layer blip late in training. *sigmoid* is
the mirror image — concentration crosses early (~6M tokens; 89% of heads eventually cross)
while neither norm signature ever does. *textinit* inherits rather than grows its
signatures: value-drain is present at 0 tokens, imported with the pretrained text-LM
weights, and concentration crosses before 1M tokens.

<figure id="fig3">
<img src="figures/fig6_sink_stripe.svg" alt="Query-by-key attention maps, top sink head per arm">
<figcaption><b>Figure 3: The sink stripe.</b> Query × key attention of the top sink head
per arm (row-normalized), early (top row) vs. final (bottom row); the cyan line marks
key = pos0 and the dotted line the image|text boundary. The pos0 stripe is <em>absent</em>
in baseline throughout, <em>total</em> in textinit (attn→pos0 = 0.62 at its final
checkpoint), and, rightmost panel, <em>already present at step 0</em> in textinit,
inherited from the text LM. sigmoid is omitted: its checkpoint dump selected heads by raw
(unnormalized) gate score and missed the arm's true top sink head (L7H3, 0.87
normalized), so a panel built from the dumped heads would visually understate its
concentration; sigmoid's concentration is established in Tables 1–2 and
Figure 2.</figcaption>
</figure>

Figure 3 shows the same story at the single-head level: the query × key attention map of
each arm's top sink head. The vertical stripe at key = pos0 (every query attending to the
first image token) is absent in *baseline* at both the early and final checkpoint, and
total in *textinit*. Most telling is the rightmost panel: **the stripe is already there at
step 0 in textinit**, imported with the text-LM weights before the model has seen a single
image. *sigmoid* is absent for the measurement reason given in the caption; its
concentration rests on Tables 1–2 and Figure 2.
Attention-entropy collapse, the text-LM literature's usual concentration correlate,
separates the same way: only *sigmoid* and *textinit* collapse (Appendix Fig. A3) —
entropy collapse tracks the concentration axis, not the norm axes.

**A positional footnote.** In *textinit*, the signatures partially decouple in *position*
as well as magnitude: in some seeds the most-attended token, the residual-norm peak, and
the value-drain minimum sit on three different tokens (e.g., attention at pos1, norm peak
and drain at pos13). We report this as a caveat on where to anchor textinit's magnitudes
(§5), not as an independent finding.
