# 6. Conclusion

We tracked concentration, value-norm drain, and a residual-norm ratio we read as a
massive-activation proxy as three separate quantities, across multimodal pretraining with a
randomly initialized decoder. Across the tested arms they need not co-move. Four levers produced
four different signature corners. The value-norm axis alone moved in three directions. On a
low-repetition run over a billion fresh tokens the proxy grew about 2.3× while concentration
never left zero, and at seed 0 the per-head relationship between the first two flipped sign
across arms. The signatures also arrived in different orders, and under text initialization they
sat on different tokens.

Text-LM work documents real interactions among these signatures, including a causal route
from massive activations to sinks and compression valleys [7]. What our results add is that
the coupling is optional. For each axis there is a lever that moves it without the others,
which extends the two-way text-only dissociations [9, 10] to a third axis and a new setting. The practical
consequence is blunt. One signature is not a proxy for the others. A model with no attention
sink can still carry a growing residual-norm asymmetry.

**Next steps.** A randomly initialized vision encoder, to isolate what the decoder
contributes to the residual-norm signal. A fresh-data run past 1B tokens. A scale-matched
gate control, to separate gating from the half-scale confound of §2.

**Reproducibility.** A self-validating probe (§2) computes all signatures on a fixed probe
batch, from logs taken every 100 steps, and the Appendix holds the per-seed tables. We will
release the code, the probe, the run configurations, the per-run logs, and the training
checkpoints on acceptance. We withhold the links for double-blind review.
