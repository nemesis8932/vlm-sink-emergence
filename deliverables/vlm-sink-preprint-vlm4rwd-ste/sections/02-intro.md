# 1. Introduction

Decoder-only transformers often allocate disproportionate attention to the first tokens of
a sequence, even when their content is uninformative. These attention sinks support
streaming inference [1] and have become a target of interpretation and mitigation [3].
In text language models, attention concentration often coincides with two other signatures:
reduced value-vector norms [4, 6] and massive activations in the residual stream [2, 5].
Their co-occurrence motivates a basic measurement question: do all three remain coupled
when a model learns from both images and text?

Existing results give reasons to distinguish the signatures. Text-model studies report
joint emergence and causal links between massive activations and attention sinks [7, 8],
but changes to normalization or value-path gradients can preserve attention sinks while
suppressing massive activations [9, 10]. These pairwise dissociations leave open how value-norm
drain varies alongside the other two quantities during multimodal pretraining.

The distinction matters when evaluating sink interventions. Reducing attention to a token
does not establish that its value norm or residual activation has changed. A study measuring
only concentration can therefore miss a persistent norm signature. In vision-language
models, sink interventions have also been evaluated for hallucination and visual grounding
[12–15]. Interpreting such interventions requires identifying which internal quantity
changes, then testing its relationship to behavior. Here we address the measurement step
by tracking all three quantities throughout training.

Multimodal pretraining lets us study this question when the first token carries visual
content. Our sequences begin with 49 image tokens and contain no BOS token, so position 0
is the first image token. Earlier multimodal studies distinguish sinks originating in
vision encoders and language decoders [11, 12]; their training-time evidence uses a
pretrained decoder and follows a single activation magnitude [11]. We instead follow all
three signatures from initialization in a model whose decoder starts from random weights,
with a text-pretrained decoder as an inheritance control.

We use a 222M-parameter nanoVLM [17] to compare softmax attention, output-gated softmax,
unnormalized sigmoid attention, and pretrained text initialization. The encoder is pretrained
and trainable in every condition. Probes every 100 steps give a joint view of the signatures
at 2-3 seeds per condition. A separate softmax run on a larger image pool extends the study
to one billion tokens at 2.39 effective visual epochs, testing whether residual-norm growth
without attention concentration persists under lower data repetition.

The experiments establish three results. First, the training conditions produce distinct
signature profiles, with value norms reduced or amplified depending on the condition
(Figure 2, Table 1). Second, in the single-seed 1B-token run, the residual-norm ratio grows
from 1.43 to 3.22 while attention concentration remains absent at every recorded probe
(Section 4.2). Third, the concentration–value-norm correlation across layer/KV-group pairs
changes sign between conditions, from +0.76 under baseline softmax to −0.79 under text
initialization at seed 0 (Section 4.3). The contribution is joint measurement of these
three signatures during multimodal pretraining, including value-norm drain beyond the
two-signature dissociations established in text models [9, 10].
