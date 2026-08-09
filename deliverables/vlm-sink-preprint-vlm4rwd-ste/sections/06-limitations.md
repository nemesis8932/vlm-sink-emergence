# 5. Limitations

**Metrics anchored on position 0.** We measure all three signatures at the first image
token. We verified this anchoring through per-position attention *mass*, not through argmax
alone. At seed 0 position 0 is the maximum-mass token in every arm: the mean and max-head
values are 0.06 and 0.17 for baseline, 0.07 and 0.23 for g1gate, and 0.30 and 0.66 for
sigmoid, and 0.63 and 0.99 for textinit, whose next-highest position reads 0.009
(Appendix C). We then scanned attention mass, residual norms and value norms per position
at the other seeds. Position 0 stays the maximum-mass token for *baseline*, *g1gate* and
*sigmoid* at every seed scanned. It does **not** in *textinit*: at seed 1 the residual peak
moves to pos1 and the value minimum to pos5 while attention stays at pos0, and at seed 2
the attention maximum sits at pos1 while the residual peak and value trough sit at pos13
(§3.4). Our pos0-anchored magnitudes for textinit at those seeds therefore *understate* the
arm's peak values, which is a further reason we report textinit as a range and a median and
treat its corner rather than its magnitudes as the claim. In *RF* the attention argmax sits
at pos1, but with mass 0.100 against 0.083 at pos0, a diffuse profile rather than a
displaced sink; the RF negative result does not depend on the anchor.

**The gate arm carries a scale confound.** We initialize the G1 gate at exactly zero, so it
opens at σ(0) = 0.5 and halves attention output at step 0, where Qiu et al. [5] use ordinary
initialization (§2, §4). Gating and initial output scaling are therefore confounded in our
*g1gate* arm, and its differences from baseline cannot be attributed to gating alone. A
scale-matched control is future work.

**Pretrained vision encoder: no arm is fully from scratch.** The SigLIP encoder is
pretrained and trainable in every arm, and *textinit* additionally uses a pretrained
decoder. What we study is therefore vision–language pretraining with randomly initialized
decoders, not from-scratch training of the whole model, and we word it that way throughout.
Vision transformers grow high-norm register tokens of their own [9], and sinks can propagate
from a vision transformer into a large vision–language model [7]. Part of our residual-norm
signal could therefore be inherited rather than decoder-formed. Our defense is the
trajectory. The h-ratio starts at 1.0–1.4 at initialization and *rises* through training,
from 1.43 to 3.22 across 1B tokens in the RF arm. Pure inheritance from a static encoder
predicts a high, flat h-ratio from step 0 instead. A control with a randomly initialized
vision transformer, which would isolate the decoder entirely, is future work.

**The h-ratio is a proxy, not a measurement of massive activations.** Massive activations
are normally defined by channel-level outliers [10, 11]. We measured a position-specific
residual-norm ratio and never computed channel-level statistics, so we report the h-ratio as
a massive-activation proxy (§2). A large h-ratio is consistent with massive activations but
does not establish them.

**Token scale.** Our runs reach at most 1B tokens per arm, against the roughly 5B tokens
that are canonical in the text-LM sink literature [1]. Sink emergence is early relative to
that budget. Text-LM sinks and their companions form near step 1000 [2], far inside our
range. We nonetheless cannot rule out that a signature absent at 1B tokens emerges later.
Confirmation at larger scale is future work.

**Reproducibility of textinit magnitudes.** The massive activation of *textinit* is
seed-sensitive. The h-ratio spans 5.5–42.5 across three seeds, and seed 0 is the consistent
outlier on every signature. The corner is the reproducible claim: high concentration, plus
strong drain, plus large massive activation. No specific magnitude is.

**Provenance and seed count.** We trust the seed-0 raw probes for the four-arm comparison
from a checksummed archive summary rather than re-derive them first-hand. We *did*
independently re-derive seeds 1 and 2. That audit caught and corrected a metric-labeling
error in an earlier internal consolidation, which mixed mean and max attention in one
column. That correction is why we report the metrics like-for-like here. *baseline* and *sigmoid*
have two seeds. *g1gate* and *textinit* have three. The RF arm uses a **single seed**, and
it contains one **weights-only optimizer restart at about 57M tokens** that an
out-of-memory error forced: the model weights were reloaded and the AdamW moment estimates
were discarded, so RF is not a single uninterrupted optimizer trajectory. The audit verified
signature continuity across that seam. The v-ratio and h-ratio values are identical at the
shared checkpoint. A double-covered 600-step overlap diverges only within probe noise, and
concentration reads 0.000 on both sides. The decoupling movement also completes before the
seam. Concentration was reproducibly zero across both seeds of the repeated-data baseline,
which we take as adequate support for the negative claim. A second fresh-data seed would
strengthen it.

**Stream-order and probe-batch caveats on RF.** Two further details of the RF run are worth
stating. The streaming shuffle buffer was reduced from 1500 to 500 examples partway through,
again for memory reasons, so stream ordering is not homogeneous across the run; the
Sink^0.3 = 0.000 result and the h-ratio rise both hold within each regime separately. And
the RF probe batch is still the **fixed repeated-`the_cauldron` tail** used by the other
arms, not a sample of the FineVision stream. That choice keeps RF's signatures comparable to
the repeated arms, which is what the negative result needs, but it means RF's signatures are
measured on data from the other distribution.

**Domain shift in the fresh-data control.** The RF arm reduces repetition by a change of
dataset, dropping the visual-epoch count from about 74 — a figure that describes the 1B-token
repeated pool, not the 100M-token comparison arms, which sit near 7 epochs — to 2.39. That
change introduces a domain-shift confound in its place. We accepted the trade deliberately.
Repetition is the dominant confound, and we chose the fresh pool to minimize shift, as
natural images, COCO-heavy. We estimate the overlap with the repeated subsets at under 3%
from the config-level composition of the two pools; we did not run image-level deduplication,
so that figure is an estimate and not measured evidence. A domain-matched control that
compares fresh and repeated data is the known follow-up.

**We report no benchmark accuracy.** We measure sink signatures with the probe of §2. We
did not run downstream benchmark evaluation, such as MMStar, on any arm. We make no claim
about how signature dissociation relates to downstream capability.
