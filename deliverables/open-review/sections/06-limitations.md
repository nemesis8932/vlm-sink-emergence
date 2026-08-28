# 5. Limitations

**Metrics anchored on position 0.** We measure all three signatures at the first image token, and
check that anchor by per-position attention *mass* rather than argmax alone. Position 0 is the
maximum-mass token in every arm at seed 0, and it stays so for *baseline*, *g1gate* and *sigmoid*
at every seed we scanned. It does **not** in *textinit*, where at seeds 1 and 2 the peaks move to
other positions (per-position table in Appendix C). The pos0-anchored textinit magnitudes at those
seeds therefore *understate* its peaks, one more reason to treat that arm's corner rather than its
magnitudes as the claim. In *RF* the attention argmax sits at pos1, but with
mass 0.053 against 0.044 at pos0, a diffuse profile rather than a displaced sink, so the RF
result does not depend on the anchor.

**The gate arm carries a scale confound.** We initialize the G1 gate at exactly zero, so it opens
at σ(0) = 0.5 and halves attention output at step 0, where Qiu et al. [20] use ordinary
initialization (§2). Gating and initial output scaling are therefore confounded in *g1gate*. We
cannot attribute its differences from baseline to gating alone, and a scale-matched control is
future work.

**Pretrained vision encoder: no arm is fully from scratch.** The SigLIP encoder is
pretrained and trainable in every arm, and *textinit* also uses a pretrained decoder. We study vision–language pretraining with randomly initialized decoders, not from-scratch training
of the whole model. Vision transformers grow high-norm register tokens of their own [25], and sinks can propagate from
a vision transformer into a vision–language model [11], so part of our residual-norm signal could
be inherited rather than decoder-formed. Our defense is the trajectory. The h-ratio starts at
1.0–1.4 and *rises* to 3.22 across 1B tokens in RF, where inheritance from a static encoder
predicts a high, flat h-ratio from step 0. A randomly initialized vision transformer, which would
isolate the decoder, is future work.

**The h-ratio is a proxy, not a measurement of massive activations.** The literature defines
massive activations by channel-level outliers [2, 5]. We measured a position-specific
residual-norm ratio and never computed channel-level statistics (§2). A large h-ratio is
consistent with massive activations without establishing them.

**Token scale.** Our runs reach at most 1B tokens per arm, against the roughly 5B canonical in the
text-LM sink literature [6]. Text-LM sinks form near step 1000 [7], far inside our range, but a
signature absent at 1B could still emerge later.

**Reproducibility of textinit magnitudes.** The massive-activation proxy of *textinit* is
seed-sensitive, an h-ratio of 5.5–42.5 across three seeds, with seed 0 the outlier on every
signature. The reproducible claim is the corner: strong concentration, strong drain, large
residual-norm ratio. No specific magnitude is.

**Provenance and seed count.** The seed-0 raw probes for the four-arm comparison come from a checksummed archive summary rather
than first-hand re-derivation. Seeds 1 and 2 we *did* re-derive independently, an audit that caught
a metric-labeling error in an earlier internal consolidation (Appendix G). *baseline* and *sigmoid* have two seeds, *g1gate* and *textinit* three, RF a **single seed**. RF
also contains one **weights-only optimizer restart at about 57M tokens**, forced by an out-of-
memory error. We reloaded the weights and discarded the AdamW moment estimates, so RF is not one
uninterrupted optimizer trajectory. The audit verified continuity across that seam: v-ratio and h-ratio identical at the shared
checkpoint, a double-covered 600-step overlap diverging only within probe noise, concentration
0.000 on both sides. Concentration was reproducibly zero across both repeated-data baseline seeds,
adequate support for the negative claim, and a second fresh-data seed would strengthen it.

**What the RF control does and does not establish.** RF has no distinct seen split, so we cannot
run for it the `val_seen` / `val_unseen` comparison that exposes memorization in the repeated arms. The weaker statement its data supports is the falling-slope statement of §2. Its streaming shuffle
buffer dropped from 1500 to 500 examples partway through, again for memory, so stream ordering is
not homogeneous, though the Sink^0.3 = 0.000 result and the h-ratio rise hold within each regime
separately. Its probe batch is still the **fixed repeated-`the_cauldron` tail** used by the other
arms. That keeps RF's signatures comparable to theirs, which is what the negative result needs, at
the cost of measuring them on data from the other distribution. And RF buys lower repetition by
changing dataset, so it trades the repetition confound for a domain-shift one, and we
chose the fresh pool to minimize shift. The under-3% overlap figure is config-level, not
image-level (§2), and the third, domain-matched control we did not run is the known follow-up.

**We report no benchmark accuracy.** We measure sink signatures only (§2): no downstream benchmark
evaluation, such as MMStar, on any arm, and no claim about how signature dissociation relates to
downstream capability.
