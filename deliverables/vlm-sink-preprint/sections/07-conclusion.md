# 6. Conclusion

We tracked the three signatures commonly bundled as "the attention sink" — concentration,
value-norm drain, and massive activation — as separate quantities across from-scratch
multimodal pretraining, and they came apart everywhere we looked. Four training levers
produced four distinct signature corners, with the value-norm axis alone moving in three
different directions; on a confound-free billion-token fresh-data run, massive activation
grew +130% while concentration never left zero; per-head correlation between concentration
and value-norm flipped sign across arms; and the signatures arrived in different orders —
norms first and concentration never in the softmax-scratch arms, the mirror image under
sigmoid attention, and everything at once, inherited at step 0, under text initialization.

None of this shows the signatures are unrelated in general: text-LM work documents real
interactions among them, including a causal route from massive activations to sinks and
compression valleys [2]. What it shows is that the coupling is not obligatory — in
from-scratch multimodal pretraining, each axis can be moved independently by ordinary
training-time choices, extending the two-way text-only dissociations [3, 4] to a third axis
and a new setting. The practical corollary for interpretability and mitigation work is that
treating any one signature as a proxy for the others is unsafe: a model with no attention
sink can still carry a growing massive activation, and a gate that removes value-drain may
leave everything else untouched.

**Next steps.** A random-initialized vision encoder to isolate the decoder's contribution
to massive activation; a per-position norm scan across all seeds to close the remaining
anchoring caveat on the text-initialized arm; and extending the fresh-data run past 1B
tokens to match text-LM budgets.

**Reproducibility.** All signatures are computed by a self-validating probe (§2) on a fixed
probe batch, from dense (every-100-step) logs; per-seed tables are in the Appendix. Code,
probe, run configurations, and checkpoints will be released with the paper.
