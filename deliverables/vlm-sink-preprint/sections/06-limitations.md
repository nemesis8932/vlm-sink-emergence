# 5. Limitations

**Pos0-anchored metrics.** All three signatures are measured at the first image token. We
verified this anchoring first-hand at seed 0 by per-position attention *mass* (not just
argmax): position 0 is the maximum-mass token in every arm — baseline 0.06 mean / 0.17
max-head, g1gate 0.07 / 0.23, sigmoid 0.30 / 0.66, and textinit 0.63 / 0.99 with the
next-highest position at 0.009 (Appendix D). That validates the reported seed-0 magnitudes,
including textinit's h-ratio of 42.5. It does not close the question at seeds 1–2:
live-probe argmax data there shows the most-attended position can migrate off pos0 in
*textinit* (one seed splits across pos1 and pos13), so part of textinit's cross-seed
magnitude spread may be position migration rather than pure magnitude noise. This is one
more reason we report textinit as a range/median and treat its corner, not its magnitudes,
as the claim. A full per-position value/residual-norm scan across all seeds is left to a
camera-ready revision.

**Pretrained vision encoder.** The SigLIP encoder is pretrained, so no arm is fully from
scratch — only the decoder is. ViTs grow high-norm register tokens of their own [9], and
sinks can propagate from a ViT into an LVLM [7], so part of our massive-activation signal
could in principle be inherited rather than decoder-formed. Our defense is the trajectory:
h-ratio starts at 1.0–1.4 at initialization and *rises* through training (RF: 1.43 → 3.22
over 1B tokens); pure inheritance from a static encoder predicts a high, flat h-ratio from
step 0. A random-initialized-ViT control, isolating the decoder entirely, is future work.

**Token scale.** Runs reach at most 1B tokens per arm, versus the ~5B canonical in the
text-LM sink literature [1]. Sink emergence is early relative to that budget — text-LM
sinks and their companions form by roughly step 1k [2], far inside our range — but we
cannot rule out that a signature absent at 1B emerges later. Larger-scale confirmation is
future work.

**Textinit magnitude reproducibility.** The *textinit* massive activation is seed-sensitive
(h-ratio 5.5–42.5 across three seeds; seed 0 the consistent outlier on every signature).
The corner — high concentration + strong drain + large massive activation — is the
reproducible claim; no specific magnitude is.

**Provenance and seed count.** Seed-0 raw probes for the four-arm comparison are trusted
from a checksummed archive summary, not re-derived first-hand; seeds 1–2 *were*
independently re-derived, and that audit caught and corrected a metric-labeling error in an
earlier internal consolidation (mean vs. max attention mixed in one column), which is why
we report the metrics like-for-like here. *baseline* and *sigmoid* have two seeds; *g1gate*
and *textinit* three. The RF run is a **single seed**, and contains one OOM-forced,
weights-only resume at ~57M tokens; the audit verified signature continuity across the seam
(identical v/h values at the shared checkpoint; a double-covered 600-step overlap diverging
within probe noise; concentration 0.000 on both sides), and the decoupling movement
completes before the seam. Concentration was reproducibly zero across both seeds of the
repeated-data baseline, which we take as adequate support for the negative claim; a second
fresh-data seed would strengthen it.

**Domain shift in the fresh-data control.** RF removes the repetition confound (~74 visual
epochs → 2.39) by switching datasets, introducing a domain-shift confound in its place. We
accepted this trade deliberately — repetition is the dominant confound, and the fresh pool
was chosen to minimize shift (natural-image, COCO-heavy, <3% overlap) — and note that the
no-sink result held in both the stronger- and weaker-shuffle regimes of the stream. A
domain-matched fresh-and-repeated control is the known follow-up.

**No benchmark accuracy is reported.** We measure sink signatures with the probe of §2. We
did not run downstream benchmark evaluation (e.g., MMStar) on any arm, and we make no claim
about how signature dissociation relates to downstream capability.
