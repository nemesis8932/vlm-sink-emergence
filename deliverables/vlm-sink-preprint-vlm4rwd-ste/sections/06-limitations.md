# 5. Limitations

**Metrics anchored on position 0.** We measure all three signatures at the first image
token, verified by per-position attention *mass* rather than argmax alone. Position 0 is the
maximum-mass token in every arm at seed 0, and stays so for *baseline*, *g1gate* and
*sigmoid* at every seed scanned. It does **not** in *textinit*, where at seed 1 the residual
peak moves to pos1 and the value minimum to pos5, and at seed 2 the attention maximum sits
at pos1 while both norm extrema sit at pos13 (Appendix C). Our pos0-anchored magnitudes for
textinit at those seeds therefore *understate* its peaks, a further reason we report that
arm as a range and a median and treat its corner rather than its magnitudes as the claim. In
*RF* the attention argmax sits at pos1, but with mass 0.100 against 0.083 at pos0, a diffuse
profile rather than a displaced sink, so the RF negative result does not depend on the
anchor.

**The gate arm carries a scale confound.** We initialize the G1 gate at exactly zero, so it
opens at σ(0) = 0.5 and halves attention output at step 0, where Qiu et al. [20] use
ordinary initialization (§2). Gating and initial output scaling are therefore confounded in
our *g1gate* arm, and its differences from baseline cannot be attributed to gating alone. A
scale-matched control is future work.

**Pretrained vision encoder: no arm is fully from scratch.** The SigLIP encoder is
pretrained and trainable in every arm, and *textinit* additionally uses a pretrained decoder.
What we study is therefore vision–language pretraining with randomly initialized decoders,
not from-scratch training of the whole model, and we word it that way throughout. Vision
transformers grow high-norm register tokens of their own [25], and sinks can propagate from
a vision transformer into a large vision–language model [11], so part of our residual-norm
signal could be inherited rather than decoder-formed. Our defense is the trajectory: the
h-ratio starts at 1.0–1.4 and *rises* to 3.22 across 1B tokens in RF, where pure inheritance
from a static encoder predicts a high, flat h-ratio from step 0. A randomly initialized
vision transformer, which would isolate the decoder entirely, is future work.

**The h-ratio is a proxy, not a measurement of massive activations.** Massive activations
are normally defined by channel-level outliers [2, 5]. We measured a position-specific
residual-norm ratio and never computed channel-level statistics, so we report the h-ratio as
a massive-activation proxy (§2). A large h-ratio is consistent with massive activations but
does not establish them.

**Token scale.** Our runs reach at most 1B tokens per arm, against the roughly 5B tokens
that are canonical in the text-LM sink literature [6]. Sink emergence is early relative to
that budget, and text-LM sinks form near step 1000 [7], far inside our range. We nonetheless
cannot rule out that a signature absent at 1B tokens emerges later.

**Reproducibility of textinit magnitudes.** The massive-activation proxy of *textinit* is
seed-sensitive, at an h-ratio spanning 5.5–42.5 across three seeds, with seed 0 the
consistent outlier on every signature. The corner is the reproducible claim: strong
concentration, plus strong drain, plus a large residual-norm ratio. No specific magnitude
is.

**Provenance and seed count.** We trust the seed-0 raw probes for the four-arm comparison
from a checksummed archive summary rather than re-derive them first-hand. We *did*
independently re-derive seeds 1 and 2, and that audit caught a metric-labeling error in an
earlier internal consolidation (Appendix G). *baseline* and *sigmoid* have two seeds,
*g1gate* and *textinit* three, and the RF arm a **single seed**. RF also contains one
**weights-only optimizer restart at about 57M tokens** that an out-of-memory error forced:
the weights were reloaded and the AdamW moment estimates discarded, so RF is not a single
uninterrupted optimizer trajectory. The audit verified continuity across that seam, where
the v-ratio and h-ratio are identical at the shared checkpoint, a double-covered 600-step
overlap diverges only within probe noise, and concentration reads 0.000 on both sides.
Concentration was reproducibly zero across both seeds of the repeated-data baseline, which
we take as adequate support for the negative claim. A second fresh-data seed would
strengthen it.

**What cannot be checked on RF.** RF has no distinct seen split, so we cannot run for it the
`val_seen` / `val_unseen` comparison that exposes memorization in the repeated arms, and the
weaker statement its data supports is that its held-out loss falls throughout and never
turns upward (§2). Its streaming shuffle buffer was also reduced from 1500 to 500 examples
partway through, again for memory reasons, so stream ordering is not homogeneous across the
run, though the Sink^0.3 = 0.000 result and the h-ratio rise both hold within each regime
separately. The RF probe batch is also still the **fixed repeated-`the_cauldron` tail** used
by the other arms. That keeps RF's signatures comparable to the repeated arms, which is what
the negative result needs, but it means they are measured on data from the other
distribution.

**Domain shift in the fresh-data control.** RF reduces repetition by a change of dataset,
which introduces a domain-shift confound in its place. We accepted the trade deliberately,
because repetition is the dominant confound, and chose the fresh pool to minimize shift. The
under-3% overlap figure is an estimate from config-level composition, not image-level
deduplication (§2), and we did not run a third, domain-matched control. That control, which
would compare fresh and repeated data at matched domain, is the known follow-up.

**We report no benchmark accuracy.** We measure sink signatures with the probe of §2. We did
not run downstream benchmark evaluation, such as MMStar, on any arm. We make no claim about
how signature dissociation relates to downstream capability.
