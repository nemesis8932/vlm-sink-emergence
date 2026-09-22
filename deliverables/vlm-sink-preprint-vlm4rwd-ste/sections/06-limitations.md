# 5. Limitations

**Measurement scope.** The h-ratio measures residual-norm asymmetry. Channel-level outlier
statistics are needed to establish massive activations in the stricter sense of [2, 5].
All headline metrics use one fixed 32-example probe batch and position 0. Their denominators
include both image and text positions. Supplementary image-only norm profiles retain the
RF asymmetry at the inspected checkpoint, while spatial scans show that textinit's
attention and norm extrema can separate (Appendix E). The reported textinit ratios are
position-specific measurements, not estimates of each run's largest possible asymmetry.

**Training controls.** The four conditions are not a factorial experiment. In g1gate,
learned gating and the initial 0.5 output scale are inseparable without a scale-matched
control. The vision encoder is pretrained and trainable in every run. Increasing residual
norms may involve adaptation of inherited visual structure as well as decoder learning.
A random-encoder control would test that contribution. Textinit also differs in prior
training and token budget, so its lower validation loss is not evidence for a superior
attention intervention.

**RF and generalization.** RF provides one seed at 1B tokens. It includes a weights-only
restart near 57M tokens and a smaller shuffle buffer later in training (Appendix B).
Changing from The Cauldron to FineVision reduces repetition but also changes the training
distribution. Its probe remains on The Cauldron, and no separate seen-image validation
split is available. The study uses one 222M architecture and 128-position sequences.
Later sink emergence, larger models and other sequence layouts remain untested.
We evaluate internal signatures, without downstream grounding or hallucination benchmarks.
