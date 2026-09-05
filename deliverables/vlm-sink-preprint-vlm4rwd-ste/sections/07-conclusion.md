# 6. Conclusion

Joint measurement during multimodal pretraining reveals distinct attention-sink signature
profiles. Under sigmoid attention, strong relative concentration accompanies amplified
value norms and little residual-norm asymmetry. Under text initialization, strong
concentration accompanies value drain and much larger residual norms. In the low-repetition
1B-token softmax run, the residual-norm ratio more than doubles while concentration stays
below every tested threshold at every recorded probe. Value-norm drain therefore adds a
separately varying third signature to the attention-versus-activation dissociations known
from text models [9, 10].

For VLM evaluation, these results motivate reporting attention concentration, value norms
and residual activations separately when assessing sink interventions. Their downstream
relevance requires behavioral evaluation. The most direct extensions are channel-level
activation measurements, a scale-matched gate control, and replication on a second
fresh-data seed.

**Reproducibility.** Appendix A specifies the probe and training recipe, and Appendix D
lists per-seed results. We will release code, configurations, logs and checkpoints upon
acceptance; identifying links are withheld for double-blind review.
