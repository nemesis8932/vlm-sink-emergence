# 6. Conclusion

The coupling is lever-dependent. Across four training levers, the three signatures land in four distinct corners: value norms are strongly drained, mildly drained, or amplified, and on a billion fresh tokens the massive-activation proxy more than doubles while concentration stays at zero. This extends two-way dissociations in text-only models [9, 10] to a third, separately measured axis in multimodal pretraining. In text LMs, massive activations can causally produce sinks [7]; our results show the coupling can also fail to form.

The practical implication is simple: across our arms, no one signature was a proxy for the others. A model without a concentration sink can still develop a growing residual-norm asymmetry, and an intervention judged on one signature may leave the others unchanged.

**Next steps.** A randomly initialized vision encoder, to isolate what the decoder
contributes to the residual-norm signal. A fresh-data run past 1B tokens. A scale-matched
gate control, to separate gating from the half-scale confound of §2.

**Reproducibility.** A self-validating probe (§2) computes all signatures on a fixed probe
batch, from logs taken every 100 steps, and the Appendix holds the per-seed tables. We will
release the code, the probe, the run configurations, the per-run logs, and the training
checkpoints on acceptance. We withhold the links for double-blind review.
