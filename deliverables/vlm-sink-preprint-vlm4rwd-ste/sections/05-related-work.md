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
its own. Neither paper is multimodal. Fesser et al. [25] come at the same question from
another direction, and their result is the closest external support for tracking the value
norm separately: they argue that one sink pattern can hide two different algorithms, an
"adaptive nop" that routes a head's update to a null token and a "broadcast" that
aggregates and redistributes global information, and that the two leave different traces —
nop sinks show negligible value norms, broadcast sinks produce low-rank outputs. On their
account gating implicitly assumes the nop mechanism and registers implicitly assume
broadcast. That work is on vision transformers rather than vision–language models, and it
diagnoses trained models rather than tracking emergence, but it independently motivates
treating the value norm as diagnostic rather than decorative. On the opposite side of the
argument, Qiu et al. [14] hold that outlier-driven rescaling by attention and residual sinks
is essential to stable training. The field is unsettled on two counts. It disputes whether these signatures
separate, and it disputes whether anyone should remove them at all.

**Multimodal sinks: mostly frozen backbones, inference time.** Luo et al. [7] identify
high-norm attention-sink tokens that originate in the vision transformer, and they separate
ViT-propagated sinks from LLM-emerged sinks. Their Appendix A.4 does track sink-dimension
magnitudes across alignment checkpoints, so a training-time view of multimodal sinks is not
without precedent; that view uses a frozen vision transformer and a pretrained language
model, and follows sink-dimension magnitude rather than the three signatures separately.
Choi et al. [8] likewise distinguish vision-sinks from language-sinks in a frozen large
vision–language model and gate them by layer. Both papers establish that multimodal sinks
have distinct vision-side and language-side origins, and our *textinit* inheritance result
fits that frame naturally. What is not yet available in that literature is dense, joint
tracking of concentration, value-norm and residual-norm as separate quantities in a decoder
that starts from random weights. Vision transformers also grow high-norm "register" tokens
of their own [9], which is why the pretrained encoder gets its own limitation in §5.

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

**The gating lever.** Qiu et al. [5] introduce the head-specific elementwise sigmoid gate on
attention output that our *g1gate* arm adapts. In text language models the gate "largely
reduces the attention score allocated to the first token and decreases massive activations"
while it improves quality. Our variant differs in one way that matters: Qiu et al. use
ordinary initialization, and we initialize the gate at exactly zero, so it opens at
σ(0) = 0.5 and applies a half-scale factor to attention output at step 0 (§2). Our arm is
therefore **Qiu-style G1 in a zero-initialized variant**, and gating and initial scaling are
confounded in it. With that caveat, what we observe in the multimodal setting with a
randomly initialized decoder is that concentration is already absent in the baseline, so the
axis on which our gated arm differs from baseline is the value-norm axis: the drain becomes
milder, from a v-ratio of 0.69–0.72 to 0.81–0.85, rather than being removed. The
residual-norm ratio is comparable in the two arms. That difference stays invisible unless
the three signatures are logged separately.

**Positioning.** The closest prior work separates at most two of the three axes, in
text-only models, through normalization or value-path interventions. The multimodal studies
work mostly on frozen models, and where one tracks alignment checkpoints [7] it follows
sink-dimension magnitude rather than the three signatures separately. Our claim is
therefore the conjunction: **dense, joint tracking of concentration, value-norm drain, and
the residual-norm ratio as separately measured quantities, in multimodal pretraining with a
randomly initialized decoder**, which adds value-norm drain as a third axis beyond the
massive-activation-vs-sink dissociations shown in text models [3, 4]. We do not claim
priority on decoupling itself, and we do not claim that prior multimodal work cannot study
emergence.
