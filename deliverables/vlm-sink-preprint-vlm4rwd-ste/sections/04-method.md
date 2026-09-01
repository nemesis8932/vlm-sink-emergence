# 3. Setup

## 3.1 Model and arms

All runs use a 222M-parameter nanoVLM [17]. A pretrained, trainable SigLIP-B/16 vision
encoder [18] feeds a decoder with the SmolLM2-135M architecture [19] through a learned
projector. The decoder has 30 layers of grouped-query attention, 9 query heads per layer
sharing 3 KV heads, so there are 270 (layer, query-head) pairs but only 90 (layer, KV-group)
value projections. §4.3 depends on that distinction. Each sequence is 49 image tokens as a
causal prefix followed by 79 left-padded text tokens, 128 in all. There is no BOS token, so
**position 0 is the first image token**.

Four training levers each target one sink-relevant mechanism. Everything else stays
byte-identical across arms.

| arm | attention | LM init | ViT init | lever precedent |
|---|---|---|---|---|
| *baseline* | softmax | random | pretrained | — |
| *g1gate* | softmax + elementwise σ-gate (zero-init, post-SDPA) | random | pretrained | G1 gating [20] |
| *sigmoid* | unnormalized sigmoid, no softmax | random | pretrained | Gu et al. [6] |
| *textinit* | softmax | pretrained SmolLM2-135M | pretrained | — (novel control) |

The decoder starts from random weights in every arm except *textinit*, which loads the
pretrained SmolLM2 decoder and imports whatever first-position structure text pretraining
built into it (§4.1). *g1gate* and *sigmoid* have established sink effects in text-only
models [20, 6]. *sigmoid* also tests the standard account of why a sink forms: softmax makes
every attention row sum to one, so a head with nothing to retrieve must park its mass
somewhere [6]. One caveat on the gate. Unlike Qiu et al. [20], we initialize it at exactly
zero, so it opens at σ(0) = 0.5 and the arm starts as a half-scale output intervention as well
as a gated one. We write "Qiu-style G1 in our zero-initialized variant" throughout (§5).

## 3.2 Data and the two training regimes

The four-arm comparison trains on four curated subsets of `the_cauldron` [21], about 146K
images, matched at about 100M tokens per arm. *textinit* stops at 60M, where its signatures
have plateaued (§4.1). *baseline* and *sigmoid* have two seeds, *g1gate* and *textinit*
three. The optimizer is AdamW with weight decay 0.1, following [6], gradient clip 1.0 and a
cosine schedule with 3% warmup (Appendix A).

Reusing a 146K-image pool for 100M tokens means high visual-epoch counts. The RF arm
(random-fresh) answers that objection. It re-trains the *baseline* recipe on a fresh
FineVision stream [22] to 1B tokens over about 4.6M natural images, at 2.39 effective visual
epochs. That is a low-repetition run rather than a repetition-free one, since examples
repeat about 2.4 times on average. We estimate the overlap between the fresh pool and the
repeated subsets at under 3%, from config-level composition rather than image-level
deduplication. RF has one seed. Swapping datasets trades the repetition confound for a
domain-shift confound (§5).

Training stays healthy in every reported run (held-out losses in Appendix A). Two cautions.
The lower *textinit* loss reflects its pretrained decoder, not the lever, so only *baseline*,
*g1gate* and *sigmoid* are equal-token, equal-initialization comparisons. And the
repeated-data arms show a large train–validation gap (`val_seen` near 0.44 against
`val_unseen` near 1.18), the overfitting signal that motivates RF. RF has no seen split. Its
held-out loss ends at 0.638 with a negative fitted slope over the second half, individual
evaluations fluctuating (§5). We did not run MMStar or any other downstream benchmark on any
arm, so this paper makes no capability claim.

## 3.3 Three signatures, tracked separately

We log the three sink symptoms the text-LM literature reports together, each at its own
granularity, following Gu et al. [6] at fixed sequence length. Every quantity below is
averaged over valid query positions and over a fixed probe batch.

*Concentration* (Sink^ε_1) is the fraction of the 270 (layer, query-head) pairs whose mean
attention to position 0 exceeds ε. We use the ε = 0.3 default of [6] and check ε ∈ {0.2,
0.4}. Cross-arm tables report the stricter ε = 0.2, which makes an absence claim harder to
pass.

*Value-norm ratio* (v-ratio) is the value norm at position 0 divided by the mean value norm
over the other valid positions, per layer and averaged over the 30 layers. Below 1 is
value-drain [4], above 1 amplification. A value vector belongs to a KV group and is shared
by 3 query heads, so the ratio rests on 90 distinct value projections, not 270 (§4.3).

*Residual-norm ratio* (h-ratio) is the residual-stream norm at position 0 divided by the
mean norm over the other positions, per layer and averaged over the 30 layers. We call it a
massive-activation proxy. The literature defines massive activations by channel-level
outliers [2, 5], which we never measured (§5).

All three anchor on position 0 by construction, and we check that choice. At seed 0,
per-position attention mass makes position 0 the maximum-mass token in every arm (Appendix
E), and §5 covers the seed-level exception. The *sigmoid* arm reports the row-normalized
attention view, which keeps concentration comparable across arms, and we also log the raw
sigmoid mass (Appendix F).

A probe recomputes all three from the live weights every 100 optimizer steps, on the same
32-sample batch for every run and seed, and validates itself against the real forward pass
on every call (Appendix A).
