# 6. Conclusion

We tracked the three signatures commonly bundled as "the attention sink" — concentration,
value-norm drain, and massive activation — as separate quantities across from-scratch
multimodal pretraining, and they came apart everywhere we looked. Four training levers
produced four distinct signature corners, with the value-norm axis alone moving in three
different directions. On a confound-free billion-token fresh-data run, massive activation
grew +130% while concentration never left zero. Per-head correlation between concentration
and value-norm flipped sign across arms. The signatures even arrived in different orders:
norms first and concentration never in the softmax-scratch arms, the mirror image under
sigmoid attention, and everything at once, inherited at step 0, under text initialization.

The signatures are plainly related in general — text-LM work documents real interactions
among them, including a causal route from massive activations to sinks and compression
valleys [2]. What our results show is that the coupling is optional. In from-scratch
multimodal pretraining, each axis moved independently under ordinary training-time levers,
extending the two-way text-only dissociations [3, 4] to a third axis and a new setting.
For interpretability and mitigation work, the practical upshot is blunt: one signature is
not a proxy for the others. A model with no attention sink can still carry a growing
massive activation, and a gate that removes value-drain may leave everything else
untouched.

**Next steps.** A random-initialized vision encoder to isolate the decoder's contribution
to massive activation; a per-position norm scan across all seeds to close the remaining
anchoring caveat on the text-initialized arm; and extending the fresh-data run past 1B
tokens to match text-LM budgets.

**Reproducibility.** All signatures are computed by a self-validating probe (§2) on a
fixed probe batch, from dense (every-100-step) logs; per-seed tables are in the Appendix.
Code, probe, run configurations, and checkpoints will be released with the paper.
