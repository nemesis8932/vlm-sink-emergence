# 1. Introduction

Decoder-only transformers develop a peculiar habit early in training: a disproportionate
share of attention mass concentrates on the first token(s) of the sequence, regardless of
content. This "attention sink" is not a curiosity — it is load-bearing for streaming
inference [12], complicates activation quantization through the extreme outliers that
accompany it [10], and has become a standing subject of interpretation and mitigation work
across modalities [13]. In text language models the sink arrives with company: attention
concentration, near-zero value-vector norms at the sink token ("value-state drain" [6]), and
abnormally large residual-stream activations ("massive activations" [10, 11]) have been
observed to emerge together — concentrating on the same few tokens [1] and co-emerging early
in training, by roughly step 1k [2]. Their consistent co-occurrence has led these signatures
to be treated, often implicitly, as facets of a single attention-sink phenomenon.

Whether they actually are one phenomenon is unsettled, and actively contested. One line of
work argues for genuine causal unity: massive activations in the residual stream
mathematically require representational compression, and ablating them eliminates both
compression valleys and sink formation [2]. A companion view holds that outlier-driven
rescaling by attention and residual sinks is *essential* to stable transformer training
[14]. Pulling the other way, two recent text-only studies dissociate pairs of these
signatures by intervention: normalization-scheme changes crush the massive-activation spike
while the sink ratio survives [3], and a value-scale intervention in from-scratch text LMs
preserves sinks while suppressing massive activations [4]. Each of these results is
text-only, and each separates at most two of the three signatures.

Vision–language models are a sharper instrument for this question than they might first
appear, for two reasons. First, the multimodal setting changes what position 0 *is*: in our
layout the sequence begins with 49 image tokens and there is no BOS — the candidate sink
token is a visual token, stripped of the special-token machinery that text-LM sink accounts
lean on. Second, existing multimodal sink studies analyze frozen, already-trained backbones
at inference time [7, 8]; they establish that vision-side and language-side sinks have
distinct origins, but a frozen model cannot answer an emergence question. Whether the three
signatures arise together, in sequence, or independently is only observable while they form
— which requires from-scratch pretraining with all three signatures logged separately,
densely, from step 0. To our knowledge no prior work does this in a multimodal model.

We do it at deliberately small scale: a 222M-parameter nanoVLM [18] — a pretrained
SigLIP-B/16 encoder [16] feeding a randomly initialized decoder with the SmolLM2-135M
architecture [17] — trained under four levers chosen to target one sink-relevant mechanism
each: standard softmax attention (*baseline*); a zero-initialized elementwise output gate on
attention, which in text LLMs removes both the sink and massive activations [5] (*g1gate*);
unnormalized sigmoid attention, which removes the softmax normalization that sinks are
argued to stem from [1] (*sigmoid*); and initializing the decoder from the pretrained
SmolLM2 text LM, importing whatever sink structure text pretraining built (*textinit*). A
validated probe logs concentration, value-norm ratio, and massive activation per (layer,
head) every 100 steps. Because the four-arm comparison reuses a small image pool at high
epoch counts, we additionally re-test the central negative result on a fresh, non-repeated
1-billion-token stream (*RF*), removing the overfitting confound.

**Contributions.**

1. **Four levers, four corners (n = 2–3 seeds/arm).** The four arms land in four distinct
   corners of (concentration × value-norm × massive-activation) space — no two arms share a
   signature triple, and the value-norm ratio alone moves in three qualitatively different
   directions (drained / unchanged / amplified) depending on the lever (Fig. 1, Table 2).
2. **Confound-free decoupling at 1B tokens.** On fresh data, massive activation rises +130%
   (h-ratio 1.43 → 3.22) over a full billion tokens while attention concentration stays at
   exactly zero for the entire run — 0% of heads ever cross the sink threshold (§4.2).
3. **No universal head-level coupling.** Per-head correlation between concentration and
   value-norm flips sign across arms (+0.67 baseline → −0.76 textinit; pooled −0.20),
   inconsistent with a single shared mechanism (§4.3).

We scope the novelty claim precisely: we are *not* the first to decouple sink signatures —
text-only work already separates massive activations from concentration via normalization
[3] and value-path [4] interventions. Our contribution is the conjunction: all three
signatures, including value-norm drain as a third independently moving axis, shown
separately controllable during from-scratch *multimodal* pretraining, a setting no prior
decoupling or co-emergence study has examined.
