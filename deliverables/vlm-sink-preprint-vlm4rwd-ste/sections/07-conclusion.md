# 6. Conclusion

We tracked the three signatures that the field bundles as "the attention sink" as separate
quantities across from-scratch multimodal pretraining. Those signatures are concentration,
value-norm drain, and massive activation. They came apart everywhere we looked. Four
training levers produced four different signature corners, and the value-norm axis alone
moved in three different directions. On a repetition-confound-free, single-seed run over a
billion fresh tokens, massive activation grew 130% while concentration never left zero. The
per-head correlation between concentration and value-norm flipped sign across arms. The
signatures even arrived in different orders. Norms came first and concentration never came
in the softmax-from-scratch arms. Sigmoid attention mirrored that order. Text
initialization delivered everything at once, inherited at step 0.

The signatures are plainly related in general. Work on text language models documents real
interactions among them, which include a causal route from massive activations to sinks and
compression valleys [2]. Our results show that the coupling is optional. In from-scratch
multimodal pretraining, each axis moved independently under ordinary training-time levers.
That extends the two-way text-only dissociations [3, 4] to a third axis and to a new
setting. For interpretability work and mitigation work the practical result is blunt. One
signature is not a proxy for the others. A model with no attention sink can still carry a
growing massive activation. A gate that removes value-drain can leave everything else
untouched.

**Next steps.** Four steps follow from this work. Train a randomly initialized vision
encoder, to isolate the contribution of the decoder to massive activation. Run a
per-position norm scan across all seeds, to close the remaining anchoring caveat on the
text-initialized arm. Extend the fresh-data run past 1B tokens, to match text-LM budgets.
Test whether the signature corner of an arm predicts hallucination or grounding behavior.
Use the benchmarks that the sink-intervention literature already uses, such as POPE, CHAIR,
and AMBER [21–23]. This paper does not test that last link. It establishes when the signatures
form, and in what combinations, which is the measurement that has to exist first.

**Reproducibility.** A self-validating probe (§2) computes all signatures on a fixed probe
batch, from dense logs taken every 100 steps. The Appendix holds the per-seed tables. We
will release the code, the probe, the run configurations, the per-run logs, and the
training checkpoints on acceptance. We withhold the links for double-blind review.
