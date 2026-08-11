# 4. Related Work

**Attention sinks and their companions in text language models.** Gu et al. [6] give the
canonical account: the sink token acts "more like key biases, storing extra attention
scores." They report small key and value norms at the sink as part of the same phenomenon,
connect it to massive residual-stream activations [2, 5], and show that unnormalized sigmoid
attention prevents sink formation in text models up to 1B parameters. Our *sigmoid* arm
builds on that. Guo et al. [4] call the concentration and value-drain coupling
"active-dormant" heads. Queipo-de-Llano, Arroyo et al. [7] make the strongest unity claim
and the only genuinely *causal* one: massive activations mathematically require
representational compression, and ablating a model's layer-0 massive activation removes both
compression valleys and sink formation. We reserve "causally unified" for that result alone.
Peng et al. [24] trace a first-position sink circuit emerging early in from-scratch text
pretraining, without separating the signatures.

**Prior decoupling results: text-only, two axes.** Two papers already separate pairs of
these signatures, and we scope our claim around them. Sun, Canziani, LeCun and Zhu [9] show
that massive activations and attention sinks are dissociable architectural artifacts: a
change of normalization scheme crushes the massive-activation spike while the sink ratio
survives. Chen and Yao [10] decouple the same pair from the opposite direction and from
scratch, where a value-scale intervention in 0.1–0.3B text models keeps the sinks and
suppresses massive activations. Neither treats value-norm drain as a third axis, and neither
is multimodal. Fesser et al. [23] give the closest external support for tracking the value
norm separately: one sink pattern can hide an "adaptive nop", which routes a head's own
update to a null token, or a "broadcast" that redistributes global information, and nop
sinks show negligible value norms where broadcast sinks produce low-rank outputs. That work
diagnoses trained vision transformers rather than tracking emergence. Qiu et al. [8] argue
the opposite case, that outlier-driven rescaling by attention and residual sinks is
essential to stable training. The field has settled neither question.

**Multimodal sinks: mostly frozen backbones, inference time.** Luo et al. [11] identify
high-norm attention-sink tokens originating in the vision transformer and separate
ViT-propagated from LLM-emerged sinks. Their Appendix A.4 tracks sink-dimension magnitudes
across alignment checkpoints, so a training-time view has precedent, using a frozen vision
transformer and a pretrained language model, and following one magnitude rather than three
signatures. Choi et al. [12] likewise distinguish vision-sinks from language-sinks in a
frozen model and gate them by layer. Both establish that multimodal sinks have distinct
vision-side and language-side origins, which our *textinit* inheritance result fits. What is
missing is dense, joint tracking of the three quantities in a decoder that starts from
random weights. Vision transformers also grow high-norm "register" tokens of their own [25],
which is why the pretrained encoder gets its own limitation in §5. A separate line ties
these signatures to hallucination and grounding failure in deployed models and intervenes at
inference time [13–16], the practical reason it matters which signature a mitigation moves.

**The gating lever.** Qiu et al. [20] introduce the head-specific elementwise sigmoid gate on
attention output that our *g1gate* arm adapts. In text models it "largely reduces the
attention score allocated to the first token and decreases massive activations" while
improving quality. Our variant differs in one way that matters: they use ordinary
initialization, we initialize at exactly zero, so the gate opens at σ(0) = 0.5 and applies a
half-scale factor at step 0 (§2), leaving gating and initial scaling confounded. With that
caveat, concentration is already absent in our baseline, so the gated arm differs on the
value-norm axis: the drain becomes milder, from 0.69–0.72 to 0.81–0.85, rather than
disappearing. That stays invisible unless the three signatures are logged separately.

**Positioning.** The closest prior work separates at most two of the three axes, in
text-only models, and the multimodal studies work mostly on frozen ones. Our claim is the
conjunction: **dense, joint tracking of concentration, value-norm drain, and the
residual-norm ratio as separately measured quantities, in multimodal pretraining with a
randomly initialized decoder**. Decoupling itself is not ours, and prior multimodal work
could study emergence if it chose to.
