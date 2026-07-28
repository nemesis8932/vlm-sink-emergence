# 5. Limitations

**Metrics anchored on position 0.** We measure all three signatures at the first image
token. We verified this anchoring first-hand at seed 0 through per-position attention
*mass*, not through argmax alone. Position 0 is the maximum-mass token in every arm. The mean and max-head values are 0.06
and 0.17 for baseline, 0.07 and 0.23 for g1gate, and 0.30 and 0.66 for sigmoid. They are
0.63 and 0.99 for textinit, whose next-highest position reads 0.009 (Appendix C). That check
validates the reported seed-0 magnitudes, which include the textinit h-ratio of 42.5. It
does not close the question at seeds 1 and 2. Live-probe argmax data at those seeds shows
that the most-attended position can migrate off position 0 in *textinit*. One seed splits
across pos1 and pos13. Part of the cross-seed magnitude spread of textinit may
therefore come from position migration rather than from magnitude noise alone. That is one
more reason we report textinit as a range and a median, and treat its corner rather than
its magnitudes as the claim. We leave a full per-position scan of value norms and residual
norms across all seeds to a camera-ready revision.

**Pretrained vision encoder.** The SigLIP encoder is pretrained, so no arm is fully from
scratch. Only the decoder is. Vision transformers grow high-norm register tokens of their
own [9], and sinks can propagate from a vision transformer into a large vision–language
model [7]. Part of our massive-activation signal could therefore be inherited rather than
decoder-formed. Our defense is the trajectory. The h-ratio starts at 1.0–1.4 at
initialization and *rises* through training, from 1.43 to 3.22 across 1B tokens in the RF
arm. Pure inheritance from a static encoder predicts a high, flat h-ratio from step 0
instead. A control with a randomly initialized vision transformer, which would isolate the
decoder entirely, is future work.

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
it contains one weights-only resume at about 57M tokens that an out-of-memory error forced.
The audit verified signature continuity across that seam. The v-ratio and h-ratio values
are identical at the shared checkpoint. A double-covered 600-step overlap diverges only
within probe noise, and concentration reads 0.000 on both sides. The decoupling movement
also completes before the seam. Concentration was reproducibly zero across both seeds of
the repeated-data baseline, which we take as adequate support for the negative claim. A
second fresh-data seed would strengthen it.

**Domain shift in the fresh-data control.** The RF arm removes the repetition confound by a
change of dataset, and drops the visual-epoch count from about 74 to 2.39. That change
introduces a domain-shift confound in its place. We accepted the trade deliberately.
Repetition is the dominant confound, and we chose the fresh pool to minimize shift, as
natural images, COCO-heavy, with less than 3% overlap. The no-sink result also held in both
the stronger-shuffle and the weaker-shuffle regimes of the stream. A domain-matched control
that compares fresh and repeated data is the known follow-up.

**We report no benchmark accuracy.** We measure sink signatures with the probe of §2. We
did not run downstream benchmark evaluation, such as MMStar, on any arm. We make no claim
about how signature dissociation relates to downstream capability.
