# 6. Conclusion

We tracked concentration, value-norm drain, and a residual-norm ratio that we read as a
massive-activation proxy as separate quantities, across multimodal pretraining with
randomly initialized decoders. They came apart everywhere we looked. Four training levers
produced four different signature corners, and the value-norm axis alone moved in three
directions. On a low-repetition, single-seed run over a billion fresh tokens, at 2.39
effective visual epochs, the massive-activation proxy grew about 2.3× while concentration
never left zero. The per-head correlation between the first two flipped sign across arms.
The signatures even arrived in different orders, and under text initialization they
separated in position as well.

Text-LM work documents real interactions among these signatures, including a causal route
from massive activations to sinks and compression valleys [7]. Our results show that the
coupling is not obligatory: each axis moved separately under ordinary training-time levers,
which extends the two-way text-only dissociations [9, 10] to a third axis and a new setting.
For interpretability and mitigation work the practical result is blunt. One signature is not
a proxy for the others. A model with no attention sink can still carry a growing
residual-norm asymmetry, and a gate that changes value-drain can leave the other axes where
they were.

**Next steps.** Train a randomly initialized vision encoder, to isolate the decoder's
contribution to the residual-norm signal. Extend the fresh-data run past 1B tokens, to match
text-LM budgets. Run a scale-matched gate control, to separate gating from the half-scale
confound of §2. Each would sharpen what this paper measures: when the signatures form, and
in what combinations.

**Reproducibility.** A self-validating probe (§2) computes all signatures on a fixed probe
batch, from dense logs taken every 100 steps. The Appendix holds the per-seed tables. We
will release the code, the probe, the run configurations, the per-run logs, and the training
checkpoints on acceptance. We withhold the links for double-blind review.
