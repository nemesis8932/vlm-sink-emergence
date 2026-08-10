# 1. Introduction

Decoder-only transformers learn a habit early in training. They send a large share of
attention to the first token or tokens of a sequence, whatever those tokens contain. The
habit has practical weight. Streaming inference methods depend on it [1]. The extreme
activation outliers that come with it make quantization harder [2]. A survey now organizes
a subfield around how to interpret the habit and how to remove it [3].

In text language models the sink does not arrive alone. Three measurements move together.
Attention concentrates on the sink token. The value vector at that token drops to a
near-zero norm, which the literature calls value-state drain [4]. The residual stream at
that token grows an abnormally large norm, which the literature calls a massive activation
[2, 5]. Gu et al. [6] report that these effects settle on the same few tokens.
Queipo-de-Llano, Arroyo et al. [7] report that they appear early in training, near step
1000. This regular co-occurrence has led the field to treat the three signatures as facets
of one attention-sink phenomenon.

The field disputes whether they really are one phenomenon. One line of work argues for
causal unity. Massive activations in the residual stream mathematically require
representational compression. Ablation of those activations removes both compression
valleys and sink formation [7]. A related view holds that outlier-driven rescaling by
attention and residual sinks is *essential* to stable transformer training [8]. Two recent
text-only studies pull the other way. Each separates a pair of these signatures by
intervention. A change of normalization scheme crushes the massive-activation spike, and
the sink ratio survives [9]. A value-scale intervention in from-scratch text language models
keeps the sink and suppresses massive activations [10]. Both results are text-only. Each one
separates at most two of the three signatures.

Vision–language models make a sharp instrument for this question, for two reasons. First,
the multimodal setting changes what position 0 *is*. Our sequence starts with 49 image
tokens and has no BOS token. The candidate sink token is therefore a visual token. It has
none of the special-token machinery that text-LM sink accounts use. Second, the multimodal
sink studies we know analyze mostly frozen, already-trained backbones at inference time
[11, 12]. They show that vision-side and language-side sinks have different origins. Luo et
al. [11] do track sink-dimension magnitudes across alignment checkpoints, with a frozen
vision transformer and a pretrained language model. A reader still cannot see, in that
setting, whether the three signatures arrive together, in sequence, or independently while
they form in a decoder that starts from random weights. That requires pretraining with a
randomly initialized decoder and all three signatures logged separately from step 0.

The question also carries deployment stakes. Attention allocation is the mechanism that
connects the language a vision–language model generates to the content of the image. A sink
is a measurable failure mode of *where* that allocation goes. A growing literature ties
these same signatures to hallucination in deployed vision–language models. Visual attention
sinks absorb attention, massive activation of specific hidden dimensions drives them, and
redistribution methods recover the absorbed attention [13]. Hallucination errors cluster in
the decoding steps immediately after the model generates a sink token, which motivates
sink-aware decoding [14]. One paper proposes a causal chain from positional encoding
through massive activations to visual sinks and then to hallucination [15]. In text
language models, sink attention together with value-norm structure carries enough signal to
detect hallucinations [16]. All of that work probes or intervenes on models that are
already trained. The question underneath it is when these signatures form during training,
and whether they form as one thing or as several. This paper answers that question. We do
not measure whether their dissociation predicts grounding or hallucination behavior. This
paper establishes the training-dynamics precondition.

We work at deliberately small scale. The model is a 222M-parameter nanoVLM [17]. A
pretrained SigLIP-B/16 encoder [18] feeds a randomly initialized decoder with the
SmolLM2-135M architecture [19]. We train it under four levers. Each lever targets one
sink-relevant mechanism. *baseline* uses standard softmax attention. *g1gate* adds a
Qiu-style G1 output gate in our zero-initialized variant, an elementwise gate on attention
output, which removes both the sink and massive activations in text language models [20].
*sigmoid* replaces softmax with unnormalized sigmoid attention. That removes the
normalization from which sinks are argued to come [6]. *textinit* initializes the decoder
from the pretrained SmolLM2 text language model, and thus imports whatever sink structure
text pretraining built. A validated probe logs concentration per query head, value-norm
ratio per KV head, and the residual-norm ratio per layer, every 100 steps. The four-arm
comparison reuses a small image pool; we therefore re-test the central negative result
under low repetition, on a fresh stream of one billion tokens (*RF*, 2.39 effective visual
epochs).

**Contributions.**

1. **Four levers, four corners (n = 2–3 seeds/arm).** The four arms reach four different
   corners of the three-signature space. No two arms
   share a signature triple. The value-norm ratio alone moves in three qualitatively
   different directions, drained or unchanged or amplified, and which one occurs differs by
   lever (Fig. 1, Table 2). The four arms are not a factorial design, so we report distinct
   intervention-associated profiles, not isolated causal effects.
2. **Low-repetition decoupling at 1B tokens (n = 1).** On a fresh stream at 2.39 effective
   visual epochs, with no observed overfit, the massive-activation proxy rises from an
   h-ratio of 1.43 to 3.22, about 2.3×, across a full billion tokens. Attention
   concentration stays at exactly zero for the entire single-seed run. No head ever crosses
   the sink threshold (§3.2, §5).
3. **No consistent-sign head-level relationship.** The per-head correlation between
   concentration and value-norm flips sign across arms (+0.67 baseline → −0.76 textinit,
   pooled −0.20). We report these descriptively (§3.3).

Text-only work already separates massive activations from concentration through
normalization [9] and value-path [10] interventions. Decoupling by itself is therefore not
new, and we do not claim it. The new part is the conjunction: dense, joint tracking of all
three signatures under randomly initialized decoders in a multimodal model, with value-norm
drain as a third axis that moves on its own.
