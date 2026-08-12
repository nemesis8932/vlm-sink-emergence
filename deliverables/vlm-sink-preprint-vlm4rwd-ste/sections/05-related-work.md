# 4. Related Work

**Attention sinks and their companions in text language models.** Gu et al. [6] give the
canonical account: the sink token acts "more like key biases, storing extra attention
scores," carries small key and value norms as part of the same phenomenon, and connects to
massive residual-stream activations [2, 5]. They also show unnormalized sigmoid attention
prevents sink formation in text models up to 1B parameters, which our *sigmoid* arm builds
on. Guo et al. [4] call the concentration and value-drain coupling "active-dormant" heads.
Queipo-de-Llano, Arroyo et al. [7] make the strongest unity claim and the only genuinely
*causal* one: massive activations mathematically require representational compression, and
ablating a model's layer-0 massive activation removes both compression valleys and sink
formation. We reserve "causally unified" for that result alone. Peng et al. [24] trace a
first-position sink circuit emerging early in from-scratch text pretraining, without
separating the signatures.

**Prior decoupling results: text-only, two axes.** Two papers already separate pairs of
these signatures, and we scope our claim around them. Sun, Canziani, LeCun and Zhu [9] show
massive activations and attention sinks are dissociable architectural artifacts: a change of
normalization scheme crushes the massive-activation spike while the sink ratio survives.
Chen and Yao [10] decouple the same pair from the opposite direction and from scratch: a
value-scale intervention in 0.1–0.3B text models keeps the sinks and suppresses massive
activations. Neither treats value-norm drain as a third axis, and neither is multimodal.
Fesser et al. [23] give the closest external support for tracking the value norm separately:
one sink pattern can hide an "adaptive nop" (negligible value norms) or a "broadcast" that
redistributes global information (low-rank outputs), though that work diagnoses trained
vision transformers rather than tracking emergence. Qiu et al. [8] argue the opposite case,
that outlier-driven rescaling by attention and residual sinks is essential to stable
training. The field has settled neither question.

**Multimodal sinks: mostly frozen backbones, inference time.** Luo et al. [11] separate
ViT-propagated from LLM-emerged sinks. Their Appendix A.4 tracks sink-dimension magnitudes
across alignment checkpoints, so a training-time view has precedent, though on a frozen
vision transformer with a pretrained language model, following one magnitude rather than
three signatures. Choi et al. [12] likewise distinguish vision-sinks from language-sinks in
a frozen model and gate them by layer. Both establish distinct vision-side and language-side
origins, which our *textinit* inheritance result fits. What is missing is dense, joint
tracking of the three quantities in a decoder that starts from random weights. Vision transformers also grow high-norm "register" tokens of their own [25], hence the pretrained-
encoder limitation in §5. A separate line ties these signatures to
hallucination and grounding failure in deployed models and intervenes at inference time
[13–16], the practical reason it matters which signature a mitigation moves.

**The gating lever.** Qiu et al. [20] introduce the head-specific elementwise sigmoid gate
on attention output that our *g1gate* arm adapts. In text models it "largely reduces the
attention score allocated to the first token and decreases massive activations" while
improving quality. Our zero-initialized variant confounds gating with initial output scaling (§2, §5). With that
caveat, concentration is already absent in our baseline, so the gated arm differs on the value-norm
axis: the drain becomes milder, 0.69–0.72 to 0.81–0.85, rather than disappearing — invisible unless
the three signatures are logged separately.

**Positioning.** The closest prior work separates at most two of the three axes, in text-only
models, and the multimodal studies work mostly on frozen ones. Our claim is the conjunction:
**concentration, value-norm drain and the residual-norm ratio tracked jointly and separately in
multimodal pretraining with a randomly initialized decoder**, adding value-norm drain as a third
axis beyond the text-only dissociations [9, 10]. Decoupling itself is not ours, and prior
multimodal work could study emergence if it chose to.
