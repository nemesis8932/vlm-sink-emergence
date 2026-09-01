# 6. Conclusion

The three signatures that define an attention sink in text models do not have to travel
together. Which of them a vision–language model develops depends on the training lever:
across four interventions, the signatures map to separate corners of signature space, with
value norms shifting between strong drain, mild drain and amplification. Most notably, over a
one-billion-token fresh stream, the massive-activation proxy more than doubles while attention
concentration stays at exactly zero. This extends prior two-way dissociations in text-only
models [9, 10] to a separately measured third axis, value-norm drain, in multimodal
pretraining. Massive activations can causally produce sinks in text LMs [7]. Our results show
the coupling can also fail to form.

**Practical implications.** The takeaway for VLM deployment and evaluation is that no single
signature acted as a reliable proxy for the others in our arms. A model without a
concentration sink can still develop a growing residual-norm asymmetry, and an intervention
judged successful by one metric may leave the other signatures unchanged. A sink mitigation
should therefore report all three.

**Status and next steps.** This is work in progress at small scale, and Section 5 names the
gaps. Next steps are a randomly initialized vision encoder to isolate the decoder's
contribution to the residual-norm signal, a second fresh-data seed and fresh-data runs past
1B tokens, a scale-matched gate control to separate gating from the half-scale initialization
confound, and channel-level statistics to turn the h-ratio proxy into a measurement of
massive activations.

**Reproducibility.** A self-validating probe (Section 3) computes all signatures on a fixed
probe batch from logs taken every 100 steps, and the Appendix holds the per-seed tables. We
will release the code, the probe, the run configurations, the per-run logs and the training
checkpoints upon acceptance. The links are withheld here for double-blind review.
