# 4. Related Work

**Attention sinks and their companions in text LMs.** Gu et al. [1] give the canonical
empirical account: the sink token acts "more like key biases, storing extra attention
scores, which could be non-informative and not contribute to the value computation" — small
key/value norms at the sink are reported as part of the same phenomenon — and the sink is
connected to massive residual-stream activations [10, 11]. They also show that removing
softmax normalization (unnormalized sigmoid attention) prevents sink formation in text LMs
up to 1B parameters at near-zero validation-loss cost, the result our *sigmoid* arm builds
on. Guo et al. [6] describe the concentration/value-drain coupling as "active-dormant"
heads, the sink head's value state actively driven toward zero. Queipo-de-Llano, Arroyo et
al. [2] make the strongest unity claim, and the only genuinely *causal* one: massive
activations mathematically require representational compression, and ablating a model's
layer-0 massive activation eliminates both compression valleys and sink formation; across
Pythia checkpoints, sinks, compression valleys, and massive activations emerge together by
roughly step 1k and stay synchronized. We reserve "causally unified" for this result
specifically. Peng et al. [15] trace a first-position sink circuit emerging early in
from-scratch text pretraining — from-scratch emergence dynamics, text-only, without
signature decoupling.

**Prior decoupling results — text-only, two axes.** Two recent papers already separate
pairs of these signatures in text models, and our claim is scoped around them. Sun,
Canziani, LeCun & Zhu [3] show massive activations and attention sinks are dissociable
architectural artifacts: switching normalization scheme (Sandwich, DynamicTanh, QKNorm vs.
Pre-Norm) crushes the massive-activation spike while the sink ratio survives — a two-way
dissociation (massive activation vs. concentration), text-only, via a normalization lever,
on trained checkpoints. Chen & Yao [4] decouple the same pair from the opposite direction
and from scratch: in 0.1–0.3B text LMs probed at dense checkpoints, a value-scale
intervention preserves sinks while suppressing massive activations. Neither treats
value-norm drain as a third, independently moving axis; neither is multimodal. On the
opposite side of the argument, Qiu et al. [14] hold that outlier-driven rescaling by
attention and residual sinks is essential to stable training — the field is actively
unsettled on whether these signatures are separable, and on whether they should be removed
at all.

**Multimodal sinks — frozen backbones, inference time.** Luo et al. [7] identify high-norm
attention-sink tokens originating in the ViT and separate ViT-propagated from LLM-emerged
sinks; Choi et al. [8] likewise distinguish vision-sinks from language-sinks in a frozen
LVLM and gate them layer-wise. Both establish that multimodal sinks have distinct vision-
and language-side origins — a frame our *textinit* inheritance result fits naturally — but
both analyze already-trained models, so neither can observe emergence, and neither measures
the three signatures as separate quantities. Relatedly, vision transformers grow high-norm
"register" tokens of their own [9], which is why our pretrained encoder is a stated
limitation (§5) rather than a footnote.

**The gating lever.** Qiu et al. [5] introduce the head-specific, zero-initialized
elementwise sigmoid gate on attention output that our *g1gate* arm uses; in text LLMs it
"largely reduces the attention score allocated to the first token and decreases massive
activations" while improving quality. Our finding sharpens what the gate does in the
multimodal from-scratch setting: with concentration already absent in the baseline, the
gate's isolated, reproducible effect is to *neutralize value-drain* (v-ratio 0.81–0.85 vs.
baseline 0.69–0.72) while leaving massive activation untouched — an effect invisible unless
the three signatures are logged separately.

**Positioning.** The closest prior work separates at most two of the three axes, in
text-only models, via normalization or value-path interventions. The multimodal studies
analyze frozen models. We are aware of no study that tracks concentration, value-norm, and
massive activation as independently measured quantities across from-scratch multimodal
pretraining. Accordingly: we claim to be **first to show all three sink signatures
independently controllable during from-scratch multimodal pretraining, adding value-norm
drain as a third axis** beyond the massive-activation-vs-sink dissociations shown in text
models [3, 4] — and we do not claim to be first to decouple sink signatures in general.
