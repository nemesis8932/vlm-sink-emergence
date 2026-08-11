# 4. Related Work

**Attention sinks and their companions in text language models.** Gu et al. [6] give the
canonical empirical account: the sink token acts "more like key biases, storing extra
attention scores, which could be non-informative and not contribute to the value
computation." They report small key and value norms at the sink as part of the same
phenomenon, connect it to massive residual-stream activations [2, 5], and show that
unnormalized sigmoid attention prevents sink formation in text models up to 1B parameters,
the result our *sigmoid* arm builds on. Guo et al. [4] describe the concentration and
value-drain coupling as "active-dormant" heads. Queipo-de-Llano, Arroyo et al. [7] make the
strongest unity claim, and the only genuinely *causal* one: massive activations
mathematically require representational compression, and ablating a model's layer-0 massive
activation removes both compression valleys and sink formation. Across Pythia checkpoints
they find the three appearing together near step 1000 and staying synchronized. We reserve
the term "causally unified" for that result alone. Peng et al. [24] trace a first-position
sink circuit emerging early in from-scratch text pretraining, in text only, without
separating the signatures.

**Prior decoupling results: text-only, two axes.** Two recent papers already separate pairs
of these signatures in text models, and we scope our claim around them. Sun, Canziani, LeCun
and Zhu [9] show that massive activations and attention sinks are dissociable architectural
artifacts: a change of normalization scheme crushes the massive-activation spike while the
sink ratio survives, a two-way dissociation in text only, through a normalization lever, on
trained checkpoints. Chen and Yao [10] decouple the same pair from the opposite direction
and from scratch, where a value-scale intervention in 0.1–0.3B text models keeps the sinks
and suppresses massive activations. Neither treats value-norm drain as a third axis that
moves on its own, and neither is multimodal. Fesser et al. [23] give the closest external
support for tracking the value norm separately: one sink pattern can hide two algorithms, an
"adaptive nop" (a no-op, where the head suppresses its own update by routing to a null
token) and a "broadcast" that redistributes global information, and nop sinks show
negligible value norms where broadcast sinks produce low-rank outputs. That work diagnoses
trained vision transformers rather than tracking emergence, but it independently motivates
treating the value norm as diagnostic rather than decorative. On the opposite side, Qiu et
al. [8] hold that outlier-driven rescaling by attention and residual sinks is essential to
stable training. The field is unsettled both on whether these signatures separate and on
whether anyone should remove them.

**Multimodal sinks: mostly frozen backbones, inference time.** Luo et al. [11] identify
high-norm attention-sink tokens that originate in the vision transformer, and separate
ViT-propagated sinks from LLM-emerged sinks. Their Appendix A.4 does track sink-dimension
magnitudes across alignment checkpoints, so a training-time view of multimodal sinks is not
without precedent. That view uses a frozen vision transformer and a pretrained language
model, and follows sink-dimension magnitude rather than the three signatures separately.
Choi et al. [12] likewise distinguish vision-sinks from language-sinks in a frozen model and
gate them by layer. Both establish that multimodal sinks have distinct vision-side and
language-side origins, which our *textinit* inheritance result fits. What is not yet
available there is dense, joint tracking of concentration, value-norm and residual-norm as
separate quantities in a decoder that starts from random weights. Vision transformers also
grow high-norm "register" tokens of their own [25], which is why the pretrained encoder gets
its own limitation in §5. A separate line ties these signatures to hallucination and
grounding failure in deployed models and intervenes at inference time [13–16], which is the
main practical reason it matters which signature a mitigation moves.

**The gating lever.** Qiu et al. [20] introduce the head-specific elementwise sigmoid gate
on attention output that our *g1gate* arm adapts. In text models it "largely reduces the
attention score allocated to the first token and decreases massive activations" while
improving quality. Our variant differs in one way that matters: Qiu et al. use ordinary
initialization, and we initialize at exactly zero, so the gate opens at σ(0) = 0.5 and
applies a half-scale factor at step 0 (§2), leaving gating and initial scaling confounded.
With that caveat, concentration is already absent in our baseline, so the axis on which the
gated arm differs is the value-norm axis: the drain becomes milder, from 0.69–0.72 to
0.81–0.85, rather than being removed. That stays invisible unless the three signatures are
logged separately.

**Positioning.** The closest prior work separates at most two of the three axes, in
text-only models, and the multimodal studies work mostly on frozen models. Our claim is
therefore the conjunction: **dense, joint tracking of concentration, value-norm drain, and
the residual-norm ratio as separately measured quantities, in multimodal pretraining with a
randomly initialized decoder**, adding value-norm drain as a third axis beyond the
massive-activation-vs-sink dissociations shown in text models [9, 10]. We do not claim
priority on decoupling itself, and we do not claim that prior multimodal work cannot study
emergence.
