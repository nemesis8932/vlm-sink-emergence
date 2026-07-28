# 4. Related Work

**Attention sinks and their companions in text language models.** Gu et al. [1] give the
canonical empirical account. The sink token acts "more like key biases, storing extra
attention scores, which could be non-informative and not contribute to the value
computation." They report small key and value norms at the sink as part of the same
phenomenon, and they connect the sink to massive residual-stream activations [10, 11]. They
also show that unnormalized sigmoid attention prevents sink formation in text language
models up to 1B parameters. Our *sigmoid* arm builds on that result. Guo et al. [6]
describe the coupling between concentration and value-drain as "active-dormant" heads,
where the model actively drives the value state of the sink head toward zero.
Queipo-de-Llano, Arroyo et al. [2] make the strongest unity claim, and the only genuinely
*causal* one. Massive activations mathematically require representational compression, and
ablation of the layer-0 massive activation of a model removes both compression valleys and
sink formation. Across Pythia checkpoints they find that sinks, compression valleys, and
massive activations appear together near step 1000 and then stay synchronized. We reserve
the term "causally unified" for that result alone. Peng et al. [15] trace a first-position
sink circuit that emerges early in from-scratch text pretraining. That work covers
from-scratch emergence dynamics, in text only, and does not separate the signatures.

**Prior decoupling results: text-only, two axes.** Two recent papers already separate pairs
of these signatures in text models, and we scope our claim around them. Sun, Canziani,
LeCun and Zhu [3] show that massive activations and attention sinks are dissociable
architectural artifacts. A change of normalization scheme crushes the massive-activation
spike, and the sink ratio survives. That is a two-way dissociation, in text only, through a
normalization lever, on trained checkpoints. Chen and Yao [4] decouple the same pair from
the opposite direction and from scratch. In text language models of 0.1–0.3B parameters,
probed at dense checkpoints, a value-scale intervention keeps the sinks and suppresses
massive activations. Neither paper treats value-norm drain as a third axis that moves on
its own. Neither paper is multimodal. On the opposite side of the argument, Qiu et al. [14]
hold that outlier-driven rescaling by attention and residual sinks is essential to stable
training. The field is unsettled on two counts. It disputes whether these signatures
separate, and it disputes whether anyone should remove them at all.

**Multimodal sinks: frozen backbones, inference time.** Luo et al. [7] identify high-norm
attention-sink tokens that originate in the vision transformer, and they separate
ViT-propagated sinks from LLM-emerged sinks. Choi et al. [8] likewise distinguish
vision-sinks from language-sinks in a frozen large vision–language model and gate them by
layer. Both papers establish that multimodal sinks have distinct vision-side and
language-side origins, and our *textinit* inheritance result fits that frame naturally. Both
papers nonetheless analyze models that are already trained. Neither can observe emergence,
and neither measures the three signatures as separate quantities. Vision transformers also
grow high-norm "register" tokens of their own [9], which is why the pretrained encoder gets
its own limitation in §5.

**Sinks and hallucination in deployed vision–language models.** A parallel line of work
ties these signatures to grounding failure. Kang et al. [21] attribute *visual* attention
sinks to massive activation of specific hidden dimensions, and they redistribute the
absorbed attention to reduce hallucination. Shukla and Kira [22] report that hallucination
errors concentrate within a few decoding steps of the generation of a sink token. They then
decode in a sink-aware way. Zhang et al. [23] trace a causal chain from rotary position
encoding through massive activations to visual sinks and then to hallucination. In text
language models, Binkowski et al. [24] detect hallucinations from sink structure. Their
classifier relies preferentially on sinks whose value vectors have large norms, which makes
it the closest precedent for treating value norms as a signal. All four papers work on
models that are already trained, at inference time. Each one treats massive activation and
concentration as one coupled mechanism, and [24] is text-only as well. None of them tracks
the signatures across training. None of them treats value-norm drain as a third axis that
moves on its own. Our result is the training-dynamics complement to that line. The coupling
on which the line depends is lever-dependent, not fixed.

**The gating lever.** Qiu et al. [5] introduce the head-specific, zero-initialized
elementwise sigmoid gate on attention output that our *g1gate* arm uses. In text language
models the gate "largely reduces the attention score allocated to the first token and
decreases massive activations" while it improves quality. Our finding sharpens what the
gate does in the from-scratch multimodal setting. Concentration is already absent in the
baseline. The isolated, reproducible effect of the gate is therefore to *neutralize
value-drain*, from a baseline v-ratio of 0.69–0.72 to 0.81–0.85. The gate leaves massive
activation untouched. That effect stays invisible unless the three signatures are logged
separately.

**Positioning.** The closest prior work separates at most two of the three axes, in
text-only models, through normalization or value-path interventions. The multimodal studies
analyze frozen models. We are aware of no study that tracks concentration, value-norm, and
massive activation as independently measured quantities across from-scratch multimodal
pretraining. Our claim is therefore the conjunction: **first to show all three sink
signatures independently controllable during from-scratch multimodal pretraining, adding
value-norm drain as a third axis** beyond the massive-activation-vs-sink dissociations
shown in text models [3, 4]. We do not claim priority on decoupling itself.
