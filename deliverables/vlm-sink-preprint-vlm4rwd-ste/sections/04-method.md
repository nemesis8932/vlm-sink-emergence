# 3. Setup

## 3.1 Model and arms

All runs use a 222M-parameter nanoVLM [17], in which a pretrained, trainable SigLIP-B/16
vision encoder [18] feeds a decoder with the SmolLM2-135M architecture [19] through a
learned projector (Fig. 1A). The decoder has 30 layers of grouped-query attention, 9 query heads per layer
sharing 3 KV heads, so there are 270 (layer, query-head) pairs but only 90 (layer, KV-group)
value projections. Section 4.3 depends on that distinction. Each sequence is 49 image tokens as a
causal prefix followed by 79 left-padded text tokens, 128 in all. There is no BOS token, so
**position 0 is the first image token**.

Four training levers each change one thing, and everything else stays byte-identical
across arms. *baseline* is plain softmax attention. *g1gate* adds an elementwise σ-gate on
the attention output, zero-initialized, after SDPA, the G1 gate of Qiu et al. [20].
*sigmoid* replaces the softmax with an unnormalized sigmoid, after Gu et al. [6]. *textinit*
keeps softmax but loads the pretrained SmolLM2-135M decoder, so it imports whatever
first-position structure text pretraining built into it (Section 4.1). The decoder starts from
random weights in every other arm, and the vision encoder is pretrained in all four. The
first two levers have established sink effects in text-only models. *textinit* has no
precedent and is an inheritance control. *sigmoid* also tests the standard account of why a sink forms: softmax makes
every attention row sum to one, so a head with nothing to retrieve must park its mass
somewhere [6]. One caveat on the gate. Unlike Qiu et al. [20], we initialize it at exactly
zero, so it opens at σ(0) = 0.5 and the arm starts as a half-scale output intervention as well
as a gated one. We write "Qiu-style G1 in our zero-initialized variant" throughout (Section 5).

## 3.2 Data and the two training regimes

The four-arm comparison trains on four curated subsets of `the_cauldron` [21], about 146K
images, matched at about 100M tokens per arm. *textinit* stops at 60M, where its signatures
have plateaued (Section 4.1). *baseline* and *sigmoid* have two seeds, *g1gate* and *textinit*
three. The optimizer is AdamW with weight decay 0.1, following [6], gradient clip 1.0 and a
cosine schedule with 3% warmup (Appendix A).

At 100M tokens the 146,731-image pool has been cycled about nine times (1.34M samples), a
high visual-epoch count. The RF arm (random-fresh) answers that objection. It re-trains the *baseline* recipe on a fresh
FineVision stream [22] to 1B tokens over about 4.6M natural images, at 2.39 effective visual
epochs. Each example is seen 2.39 times on average, so RF is low-repetition rather than
repetition-free. We estimate the overlap between the fresh pool and the
repeated subsets at under 3%, from config-level composition rather than image-level
deduplication. RF has one seed. Swapping datasets trades the repetition confound for a
domain-shift confound (Section 5).

Training stays healthy in every reported run (held-out losses in Appendix A). Two cautions.
The lower *textinit* loss reflects its pretrained decoder, not the lever, so only *baseline*,
*g1gate* and *sigmoid* are matched on tokens and initialization, differing in the attention
lever alone. And the
repeated-data arms show a large train–validation gap (`val_seen` near 0.44 against
`val_unseen` near 1.18), the overfitting signal that motivates RF. RF has no `val_seen` split,
since at 2.39 visual epochs its loader would re-use the held-out pool, so we report
`val_unseen` only. It ends at 0.638 with a negative fitted slope over the second half,
individual evaluations fluctuating (Section 5).

## 3.3 Three signatures, tracked separately

We log the three sink symptoms the text-LM literature reports together, each at its own
granularity, following Gu et al. [6] at fixed sequence length. Let $a_{l,h}$ be the mean
attention that head $(l,h)$ sends to position 0, and $\|v^{(l)}_i\|$ and $\|h^{(l)}_i\|$
the value-vector and residual-stream norms at position $i$ in layer $l$, all averaged over
valid query positions and the fixed probe batch. An overline denotes the mean over the other
valid positions $i > 0$. The decoder has $L = 30$ layers, $H = 9$ query heads and $G = 3$ KV
groups.

*Concentration* is the share of (layer, query-head) pairs whose attention to position 0
exceeds a threshold, the metric of [6],

$$
\mathrm{Sink}^{\epsilon}_1 = \frac{1}{LH}\sum_{l=1}^{L}\sum_{h=1}^{H} \mathbf{1}\!\left[a_{l,h} > \epsilon\right].
$$

We use their $\epsilon = 0.3$ default and check $\epsilon \in \{0.2, 0.4\}$. Cross-arm
tables report the stricter $\epsilon = 0.2$, which makes an absence claim harder to pass.

*Value-norm ratio* compares the value norm at position 0 with the rest of the sequence,

$$
\text{v-ratio} = \frac{1}{L}\sum_{l=1}^{L} \frac{\|v^{(l)}_0\|}{\overline{\|v^{(l)}_{i}\|}},
$$

below 1 under value-drain [4] and above 1 under amplification. A value vector is shared by
the 3 query heads of its KV group, so the ratio rests on $LG = 90$ distinct projections, not
270 (Section 4.3).

*Residual-norm ratio* is the same ratio on the residual stream,

$$
\text{h-ratio} = \frac{1}{L}\sum_{l=1}^{L} \frac{\|h^{(l)}_0\|}{\overline{\|h^{(l)}_{i}\|}}.
$$

We call it a massive-activation proxy, since the literature defines massive activations by
channel-level outliers [2, 5], which we never measured (Section 5).

All three anchor on position 0 by construction, and we check that choice. At seed 0,
per-position attention mass makes position 0 the maximum-mass token in every arm (Appendix
E), and Section 5 covers the seed-level exception. The *sigmoid* arm reports the row-normalized
attention view, which keeps concentration comparable across arms, and we also log the raw
sigmoid mass (Appendix F).

A probe recomputes all three from the live weights every 100 optimizer steps, on the same
32-sample batch for every run and seed, and validates itself against the real forward pass
on every call (Appendix A).
