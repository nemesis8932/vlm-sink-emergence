# 2. Setup

**Model and token layout.** All runs use a 222M-parameter nanoVLM [17]. A pretrained,
trainable SigLIP-B/16 vision encoder [18] feeds a decoder with the SmolLM2-135M
architecture [19] through a learned modality projector. The decoder has 30 layers of
grouped-query attention. Each layer has 9 query heads that share 3 KV heads. The decoder
therefore has 270 (layer, query-head) pairs but only 90 (layer, KV-group) value
projections. This distinction matters when we count per-head observations (§3.3). The
decoder trains **from random initialization** in every arm except *textinit*, because the
point is to watch the signatures form. Each sequence holds 49 image tokens as a causal
prefix, then 79 left-padded text tokens, for 128 tokens in all. **Position 0 is the first
image token, and there is no BOS token.** What happens at position 0 is therefore a property
of the visual prefix. It is not inherited BOS machinery.

**Arms.** We use four training levers. Each lever targets one sink-relevant mechanism.
Everything else stays byte-identical across arms.

| arm | attention | LM init | ViT init | lever precedent |
|---|---|---|---|---|
| *baseline* | softmax | random | pretrained | — |
| *g1gate* | softmax + elementwise σ-gate (zero-init, post-SDPA) | random | pretrained | G1 gating [20] |
| *sigmoid* | unnormalized sigmoid, no softmax | random | pretrained | Gu et al. [6] |
| *textinit* | softmax | pretrained SmolLM2-135M | pretrained | — (novel control) |

The *g1gate* lever and the *sigmoid* lever have established sink effects in text-only
models [20, 6]. The *textinit* lever has no precedent in the sink literature. It works as an
inheritance control. It imports whatever sink structure text pretraining already built into
SmolLM2.

**A scale confound in our gate variant.** Qiu et al. [20] use ordinary initialization for
the G1 gate. We initialize the gate parameters at exactly zero, so the sigmoid opens at
σ(0) = 0.5 at step 0. Our gated arm therefore begins as a half-scale attention-output
intervention as well as a gating one, and the two effects are not separated here. We call
the arm **Qiu-style G1 in our zero-initialized variant** throughout, and we repeat the
caveat in §5. Any comparison to the published G1 result carries it.

**Data and the two training regimes.** The four-arm comparison trains on four curated
subsets of `the_cauldron` [21], which hold about 146K images. The arms are matched at about
100M tokens each. *textinit* stops at 60M tokens, because it reaches its three-signature
floor earlier (§3.1). Reuse of a 146K-image pool gives high visual-epoch counts. A reader
could therefore treat a "no sink emerges" result as an overfitting artifact. The **RF** arm
(random-fresh) answers that objection. It re-trains the *baseline* recipe on a fresh
FineVision stream [22] to 1B tokens. That stream holds about 4.6M natural images and gives
**2.39 effective visual epochs**, against about 74 epochs for the 1B-token repeated pool.
The 100M-token comparison arms sit at roughly 7 epochs, so the 74-epoch figure describes
the repeated pool at 1B tokens, not those arms. RF is therefore a **low-repetition** run,
not a repetition-free one: examples do repeat, about 2.4 times on average. What we observe
is that no overfit accompanies it: held-out fresh validation loss tracks train loss (§2,
recipe). A change of dataset also trades the repetition confound for a domain-shift
confound. We accept that trade and document it. The fresh pool holds natural images and
leans heavily on COCO. We estimate its overlap with the repeated subsets at under 3% from
the config-level composition of the two pools; we did not run image-level deduplication, so
that number is an estimate, not measured evidence. We did not run a third, domain-matched
control (§5).

**Three signatures, tracked separately.** We log the three sink symptoms that the text-LM
literature reports together, each at its own granularity, following Gu et al. [6] at a
fixed sequence length. The decoder has *L* = 30 layers, *H* = 9 query heads per layer, and
*G* = 3 KV heads per layer. Every quantity below is averaged over valid query positions and
over the fixed probe batch.

*Concentration* (Sink^ε_1) is the fraction of the *L·H* = 270 (layer, query-head) pairs
whose mean attention to position 0 exceeds ε. We use the ε = 0.3 default of [6] and check
ε ∈ {0.2, 0.4}. Cross-arm tables report the stricter ε = 0.2, which makes an absence claim
harder to pass.

*Value-norm ratio* (v-ratio) is the value norm at position 0 divided by the mean value norm
over the other valid positions, computed per layer and then averaged over the 30 layers.
Below 1 is value-drain [4]; above 1 is amplification. Under grouped-query attention a value
vector belongs to a KV group and repeats across 3 query heads, so the ratio rests on
*L·G* = 90 independent value projections, not 270. This matters for §3.3.

*Residual-norm ratio* (h-ratio) is the residual-stream norm at position 0 divided by the
mean norm over the other positions, again per layer and then averaged over the 30 layers.
We call it a **massive-activation proxy**. Massive activations are normally defined by
channel-level outliers [2, 5], which we never measured; the h-ratio captures a
position-specific residual-norm asymmetry, necessary but not sufficient for that definition.

**Why a sink is expected at all.** Softmax forces every attention row to sum to one. A head
with nothing informative to retrieve must still place its mass somewhere. Gu et al. [6]
argue that the sink token absorbs that surplus and acts "more like key biases." That
sum-to-one constraint is the standing mechanism against which our *sigmoid* arm, which
removes normalization, is the direct test.

All three metrics **anchor on position 0 by construction**. We state this as a measurement
choice, and we check it. At seed 0, per-position attention mass makes position 0 the
maximum-mass token in every arm (Appendix C). Section 5 handles the remaining seed-level
caveat. The *sigmoid* arm reports the row-normalized attention view, which keeps
concentration comparable across arms. We also log the raw sigmoid mass.

**Probe.** The function `probe_sinks()` re-walks the decoder from the live module weights
in eager mode, in fp32, with no gradients. It is independent of the training path, which
uses fused SDPA kernels, autocast, and `torch.compile`. Every call validates its hidden
states against the real forward pass of the model, to a relative error below 10<sup>−2</sup>. The
probe therefore cannot drift from what the trained model computes. The probe batch stays
fixed across all runs and all seeds, so every number in this paper is comparable from run
to run. Probes fire every 100 optimizer steps. That cadence is dense enough to timestamp
the first threshold crossing of each signature (§3.4).

**Recipe.** We use AdamW with weight decay 0.1, following [6], and a gradient clip of 1.0.
The schedule is cosine with 3% warmup. The learning rates are 4 × 10<sup>−4</sup> for the language
model, 2 × 10<sup>−3</sup> for the projector, and 10<sup>−4</sup> for the vision encoder. We use bf16 autocast,
`torch.compile`, and a batch size of 128. Arms in a comparison differ only in the lever
under test. Training stays healthy in all reported runs.

**Reporting details.** *Token accounting:* one token is one image token or one non-padding
text token. A full sequence contributes 49 image tokens plus its non-padding text tokens, so
a step of batch 128 contributes at most 16,384 tokens and about 9.5K on average. All "M
tokens" figures in this paper use that definition. *Probe:* the fixed probe batch holds
n = 32 samples, drawn with `random.seed(0)` from the repeated `the_cauldron` tail; we label
this probe version `v1-repeatedtail-32`. RF uses the same probe batch as the repeated arms,
so its signatures stay comparable to theirs even though its training stream differs (§5).
Probes fire every 100 steps. *Validation:* 1,024 held-out examples, evaluated every 500
steps; we report `val_unseen`, which holds out images, and also log `val_seen`, which
re-uses training images with fresh question–answer text. *Seeds:* two for *baseline* and
*sigmoid*, three for *g1gate* and *textinit*, one for *RF*. *Aggregation:* the three
formulas above; each per-layer or per-head quantity is first averaged over the probe batch
and over valid query positions, then aggregated as written.

**Validation losses, and what they do and do not license.** At the matched 100M-token
checkpoint the held-out losses are 1.182 for *baseline*, 1.133 for *g1gate*, 1.206 for
*sigmoid*, and 0.877 for *textinit* (seed 1 or 2; seed-0 archive values are 1.161, 1.138,
1.108, and 0.832). RF reaches 0.638 at 1B tokens. Two cautions. First, *textinit* starts
from a pretrained text decoder, so its lower loss reflects unequal competence, not a lever
effect; only *baseline*, *g1gate*, and *sigmoid* are equal-token, equal-initialization
comparisons. Second, the repeated-data arms show a large train–validation asymmetry
(`val_seen` near 0.44 against `val_unseen` near 1.18), which is the overfitting signal that
motivates RF. RF shows no such gap: `val_seen` 0.641 against `val_unseen` 0.638. **We did
not run MMStar or any other downstream benchmark on any arm**, so this paper makes no
capability claim (§5).
