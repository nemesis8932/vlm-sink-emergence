# 5. Discussion and limitations

Across the arms the three signatures land in different corners, and no single one predicts
the others. For VLM work the consequence is direct. An intervention judged on concentration
alone can leave value-norm drain or the residual-norm ratio where they were, and a model with
no concentration sink can still grow a residual-norm asymmetry, as RF does. What that licenses
depends on the limits below. Appendix B gives the full account of each.

*The h-ratio is a proxy.* The literature defines massive activations by channel-level outliers
[2, 5]. We measured a position-specific residual-norm ratio and never computed channel-level
statistics (§3). A large h-ratio is consistent with massive activations without establishing
them.

*RF is one seed and one run.* The 1B-token result rests on a single seed with one weights-only
optimizer restart at about 57M tokens, the weights reloaded and the AdamW moments discarded.
The audit verified continuity across that seam (Appendix B). Concentration was reproducibly
zero across both repeated-data baseline seeds, which supports the negative claim, and a second
fresh-data seed would strengthen it. RF also has no seen split, and it buys low repetition by
changing dataset, so it trades the repetition confound for a domain-shift one. The under-3%
overlap figure is config-level, not image-level.

*The gate arm carries a scale confound.* Zero-initializing the G1 gate opens it at σ(0) = 0.5
and halves attention output at step 0, where Qiu et al. [20] use ordinary initialization.
Gating and initial output scaling are confounded in *g1gate*, and a scale-matched control is
future work.

*No arm is fully from scratch, and the metrics anchor on position 0.* The SigLIP encoder is
pretrained in every arm, and vision transformers grow high-norm tokens of their own [25], so
part of the residual-norm signal could be inherited. Our defense is the trajectory. The
h-ratio starts at 1.0–1.4 and rises to 3.22 across 1B tokens in RF, where a static inherited
source predicts a high, flat value from step 0. Position 0 is the maximum-mass token in every
arm at seed 0 and at every seed for the random-decoder arms, but not for *textinit* at seeds
1 and 2, where the peaks move (Appendix E). Those magnitudes understate the arm's peaks, so we
report *textinit* as a corner and a range, 5.5–42.5×, and not as a magnitude.

*Scale and scope.* Runs reach at most 1B tokens against roughly 5B canonical in the text-LM
sink literature [6]. Text-LM sinks form near step 1000 [7], well inside our range, but a
signature absent at 1B could still emerge later. The seed-0 probes for the four-arm
comparison come from a checksummed archive summary, and seeds 1 and 2 were re-derived
first-hand (Appendix B.5, Appendix H). We ran no downstream benchmark on any arm and make no capability
claim.
