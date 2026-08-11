# 1. Introduction

Decoder-only transformers pick up a habit early in training. They send a large share of
attention to the first token or tokens of a sequence, whatever those tokens contain. The
habit carries practical weight: streaming inference depends on it [1], the extreme
activation outliers that ride along with it make quantization harder [2], and a survey now
organizes a subfield around interpreting and removing it [3].

The sink does not arrive alone. Three measurements move together in text language models.
Attention concentrates on the sink token. The value vector there drops to a near-zero norm,
called value-state drain [4]. The residual stream there grows an abnormally large norm,
called a massive activation [2, 5]. Prior work reports that these effects settle on the same
few tokens [6] and appear near step 1000 [7]. The field has taken the co-occurrence as
licence to treat all three as facets of one attention-sink phenomenon.

Whether they really are one phenomenon is contested. One line argues for causal unity:
massive activations mathematically require representational compression, and ablating them
removes both compression valleys and sink formation [7]. A related view holds that
outlier-driven rescaling by attention and residual sinks is *essential* to stable training
[8]. Two recent studies pull the other way, each separating a pair of the signatures by
intervention, one through the normalization scheme [9] and one through value scale [10].
Both are text-only, and each separates at most two of the three.

Which signature a study measures therefore decides what it can conclude. If the three move
independently, a lever that clears concentration while leaving the value path alone reads as
a fix under one metric and a failure under the next, and two papers can disagree about
whether the sink was removed while both are right. The stakes reach past bookkeeping:
sink-like attention has been tied to hallucination and weak visual grounding in deployed
vision–language models [13–16]. We measure all three at once, throughout training, and test
no downstream behavior here.

Vision–language models make a sharp instrument for the question. The multimodal setting
changes what position 0 *is*: our sequence starts with 49 image tokens and has no BOS token,
so the candidate sink token is a visual token, without the special-token machinery that
text-LM sink accounts lean on. The multimodal sink studies we know analyze mostly frozen,
already-trained backbones at inference time [11, 12], and the one training-time view follows
a single magnitude rather than the three signatures (§4). Seeing whether the signatures
arrive together, in sequence, or independently takes pretraining with a randomly initialized
decoder and all three logged separately from step 0.

We work at deliberately small scale. The model is a 222M-parameter nanoVLM [17], where a
pretrained SigLIP-B/16 encoder [18] feeds a randomly initialized decoder with the
SmolLM2-135M architecture [19]. Four levers each target one sink-relevant mechanism (§2):
standard softmax attention (*baseline*), a Qiu-style G1 output gate in our zero-initialized
variant [20] (*g1gate*), unnormalized sigmoid attention, which removes the normalization
sinks are argued to come from [6] (*sigmoid*), and initialization from the pretrained
SmolLM2 text model (*textinit*). A validated probe logs all three signatures every 100
steps. The four-arm comparison reuses a small image pool, so we re-test the central negative
result under low repetition, on a fresh stream of one billion tokens (*RF*, 2.39 effective
visual epochs).

**Contributions.**

1. **Four levers, four corners (n = 2–3 seeds/arm).** The four arms reach four different
   corners of the three-signature space, and no two share a triple. The value-norm ratio
   alone is strongly drained, mildly drained, or amplified, depending on the lever (Fig. 1,
   Table 1). The arms are not a factorial design, so we report distinct intervention-associated
   profiles, not isolated causal effects.
2. **Low-repetition decoupling at 1B tokens (n = 1).** On a fresh stream at 2.39 effective
   visual epochs, with a held-out loss that never turns upward, the massive-activation proxy
   rises from an h-ratio of 1.43 to 3.22, about 2.3×. Concentration stays at exactly zero
   for the entire single-seed run, and no head ever crosses the sink threshold (§3.2, §5).
3. **No consistent-sign head-level relationship.** The per-head correlation between
   concentration and value-norm flips sign across arms (+0.76 baseline → −0.79 textinit,
   pooled −0.20, over 90 KV groups per arm). We report these descriptively (§3.3).

Text-only work already separates massive activations from concentration [9, 10], so
decoupling itself is not new and we do not claim it. The new part is the conjunction: dense,
joint tracking of all three signatures under randomly initialized decoders in a multimodal
model, with value-norm drain as a third axis that moves on its own.
