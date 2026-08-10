# 6. Conclusion

We tracked the three signatures that the field bundles as "the attention sink" as separate
quantities across multimodal pretraining with randomly initialized decoders. Those
signatures are concentration, value-norm drain, and a residual-norm ratio that we read as a
massive-activation proxy. They came apart everywhere we looked. Four training levers
produced four different signature corners, and the value-norm axis alone moved in three
different directions. On a low-repetition, single-seed run over a billion fresh tokens, at
2.39 effective visual epochs and with a held-out loss that never turned upward, the
massive-activation proxy grew
about 2.3× while concentration never left zero. The per-head correlation between
concentration and value-norm flipped sign across arms. The signatures even arrived in
different orders. Norms came first and concentration never came in the softmax arms with
random decoders. Sigmoid attention mirrored that order. Text initialization imported the
norm signatures at step 0 and crossed concentration shortly after. In that arm the
signatures separated in position as well, sitting on different tokens at two of three seeds.

The signatures are plainly related in general. Work on text language models documents real
interactions among them, which include a causal route from massive activations to sinks and
compression valleys [7]. Our results show that the coupling is not obligatory. In
multimodal pretraining with a randomly initialized decoder, each axis moved separately under
ordinary training-time levers. That extends the two-way text-only dissociations [9, 10] to a
third axis and to a new setting. For interpretability work and mitigation work the practical
result is blunt. One signature is not a proxy for the others. A model with no attention sink
can still carry a growing residual-norm asymmetry. A gate that changes value-drain can leave
the other axes where they were.

**Next steps.** Three steps follow from this work. Train a randomly initialized vision
encoder, to isolate the contribution of the decoder to the residual-norm signal. Extend the
fresh-data run past 1B tokens, to match text-LM budgets. Test whether the signature corner
of an arm predicts hallucination or grounding behavior.
Use the benchmarks that the sink-intervention literature already uses, such as POPE, CHAIR,
and AMBER [13–15]. This paper does not test that last link. It establishes when the signatures
form, and in what combinations, which is the measurement that has to exist first.

**Reproducibility.** A self-validating probe (§2) computes all signatures on a fixed probe
batch, from dense logs taken every 100 steps. The Appendix holds the per-seed tables. We
will release the code, the probe, the run configurations, the per-run logs, and the
training checkpoints on acceptance. We withhold the links for double-blind review.
