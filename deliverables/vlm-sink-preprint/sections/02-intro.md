# 1. Introduction

Early in training, decoder-only transformers start parking a disproportionate share of
attention on the first token(s) of a sequence, whatever the content. The habit matters in
practice: streaming inference schemes rely on it [12], and the extreme activation outliers
that ride along with it complicate quantization [10]; a survey now organizes a subfield
around interpreting and mitigating it [13]. In text language models the sink also arrives
with company. Attention concentration, near-zero value-vector norms at the sink token
("value-state drain" [6]), and abnormally large residual-stream activations ("massive
activations" [10, 11]) have been observed to emerge together, concentrating on the same few
tokens [1] and co-emerging early in training, by roughly step 1k [2]. Their consistent
co-occurrence has led these signatures to be treated, often implicitly, as facets of a
single attention-sink phenomenon.

Whether they actually are one phenomenon is unsettled, and actively contested. One line of
work argues for genuine causal unity: massive activations in the residual stream
mathematically require representational compression, and ablating them eliminates both
compression valleys and sink formation [2]. A companion view holds that outlier-driven
rescaling by attention and residual sinks is *essential* to stable transformer training
[14]. Pulling the other way, two recent text-only studies dissociate pairs of these
signatures by intervention. Normalization-scheme changes crush the massive-activation spike
while the sink ratio survives [3]; a value-scale intervention in from-scratch text LMs
preserves sinks while suppressing massive activations [4]. Each of these results is
text-only, and each separates at most two of the three signatures.

Vision–language models turn out to be a sharp instrument for this question, for two
reasons. First, the multimodal setting changes what position 0 *is*. In our layout the
sequence begins with 49 image tokens and there is no BOS, so the candidate sink token is a
visual token, stripped of the special-token machinery that text-LM sink accounts lean on.
Second, existing multimodal sink studies analyze frozen, already-trained backbones at
inference time [7, 8]. They establish that vision-side and language-side sinks have
distinct origins, but a frozen model cannot answer an emergence question. Whether the three
signatures arise together, in sequence, or independently is only observable while they
form, and that requires from-scratch pretraining with all three logged separately, densely,
from step 0. To our knowledge no prior work does this in a multimodal model.

We do it at deliberately small scale: a 222M-parameter nanoVLM [18] — a pretrained
SigLIP-B/16 encoder [16] feeding a randomly initialized decoder with the SmolLM2-135M
architecture [17] — trained under four levers chosen to target one sink-relevant mechanism
each. *baseline* uses standard softmax attention. *g1gate* adds a zero-initialized
elementwise output gate on attention, which in text LLMs removes both the sink and massive
activations [5]. *sigmoid* replaces softmax with unnormalized sigmoid attention, removing
the normalization that sinks are argued to stem from [1]. *textinit* initializes the
decoder from the pretrained SmolLM2 text LM, importing whatever sink structure text
pretraining built. A validated probe logs concentration, value-norm ratio, and massive
activation per (layer, head) every 100 steps. Because the four-arm comparison reuses a
small image pool at high epoch counts, we re-test the central negative result on a fresh,
non-repeated 1-billion-token stream (*RF*), removing the overfitting confound.

**Contributions.**

1. **Four levers, four corners (n = 2–3 seeds/arm).** The four arms land in four distinct
   corners of (concentration × value-norm × massive-activation) space. No two arms share a
   signature triple, and the value-norm ratio alone moves in three qualitatively different
   directions (drained / unchanged / amplified) depending on the lever (Fig. 1, Table 2).
2. **Confound-free decoupling at 1B tokens.** On fresh data, massive activation rises +130%
   (h-ratio 1.43 → 3.22) over a full billion tokens while attention concentration stays at
   exactly zero for the entire run; 0% of heads ever cross the sink threshold (§3.2).
3. **No universal head-level coupling.** Per-head correlation between concentration and
   value-norm flips sign across arms (+0.67 baseline → −0.76 textinit; pooled −0.20),
   which a single shared mechanism cannot produce (§3.3).

Text-only work already separates massive activations from concentration, via normalization
[3] and value-path [4] interventions, so decoupling per se is not new and we do not claim
it. What is new is the conjunction: all three signatures, including value-norm drain as a
third independently moving axis, shown separately controllable during from-scratch
*multimodal* pretraining — a setting no prior decoupling or co-emergence study has
examined.
