# 3. Setup

## 3.1 Model and training conditions

All runs use a 222M-parameter nanoVLM [17]. A pretrained, trainable SigLIP-B/16 encoder [18]
feeds a SmolLM2-135M-architecture decoder [19] through a learned projector (Figure 1).
The decoder has $L=30$ layers, $H=9$ query heads per layer and $G=3$ KV groups, giving
270 layer/query-head pairs and 90 distinct value projections. Each sequence contains
49 image tokens followed by 79 left-padded text positions. There is no BOS token;
position 0 is the first image token.

<figure id="fig0">
<img src="figures/fig0_overview.tex" alt="Overview: pipeline, position 0 and the probe">
<figcaption><b>Figure 1: Model and measurement setup.</b> The image encoder and projector
produce a 49-token visual prefix. The decoder receives this prefix followed by text.
Position 0 is the first image token. Probes measure attention concentration, projected
value norms and residual-stream norms every 100 steps. Decoder initialization is random
except in <em>textinit</em>; RF uses the baseline architecture on a larger image pool.</figcaption>
</figure>

The four conditions share the architecture and training recipe except for the specified
attention operation or decoder initialization. *baseline* uses softmax attention.
*g1gate* adds the head-specific elementwise sigmoid gate of Qiu et al. [20] after attention
aggregation and before the output projection. Our gate weights start at zero, making its
initial output multiplier $\sigma(0)=0.5$. *sigmoid* replaces softmax with unnormalized
sigmoid attention [6]. *textinit* retains softmax and loads pretrained SmolLM2-135M decoder
weights as an inheritance control. All other decoders start from random weights.
The gated comparison therefore includes both learned gating and initial output scaling.

## 3.2 Data and training

The four-condition comparison uses the VQAv2, COCO-QA, A-OKVQA and VSR subsets of
The Cauldron [21], containing 146,731 images. Table 1 compares checkpoints near 100M
tokens, or 60M for *textinit*; full trajectories retain the available later checkpoints.
*baseline* and *sigmoid* have two seeds each, and *g1gate* and *textinit* have three.

At 100M tokens, training has sampled this image pool about nine times. We also train
the baseline architecture on a larger FineVision stream [22], about 4.6M images, to
1B tokens at 2.39 effective visual epochs. This random-fresh run, *RF*, has one seed.
It tests persistence under lower repetition, with dataset shift considered in Section 5.
Token budgets count image tokens plus non-padding text tokens.

Training uses AdamW, weight decay 0.1, gradient clipping at 1.0, and a cosine learning-rate
schedule with 3% warmup. Appendix A gives learning rates, precision, batches and validation
splits; Appendix B records RF's optimizer restart and data-ordering changes.

## 3.3 Signature definitions

Every 100 optimizer steps, we evaluate the same 32-example probe batch and validate the
probe against the model's forward pass (Appendix A). Let $a_{l,h}$ be attention to position 0,
averaged over all non-padding query positions except position 0 itself, pooled across the
batch. Sigmoid weights are row-normalized for this diagnostic, following [6]; the model's
forward pass remains unnormalized. We report raw sigmoid weights separately in Appendix F.

*Concentration* is the fraction of query heads exceeding a threshold,

$$
\mathrm{Sink}^{\epsilon}_1=\frac{1}{LH}\sum_{l=1}^{L}\sum_{h=1}^{H}
\mathbf{1}[a_{l,h}>\epsilon].
$$

We use $\epsilon=0.3$ [6] and check 0.2 and 0.4. Table 1 uses 0.2, a lower threshold
that makes the absence of concentration harder to establish. Figure 2 instead uses
$a_{\max}=\max_{l,h}a_{l,h}$, the continuous maximum attention across all 270 pairs.

*Norm ratios* compare position 0 with the remaining valid image and text positions.
For sample $b$, position $i$, layer $l$ and KV group $g$, define
$m^{(l)}_{b,i}=G^{-1}\sum_{g=1}^{G}\|v^{(l,g)}_{b,i}\|_2$ and
$r^{(l)}_{b,i}=\|h^{(l)}_{b,i}\|_2$. Values are measured after the normalized input's value
projection; residuals are measured at the block input. Let $\langle\cdot\rangle_0$ denote
the batch mean at position 0 and $\langle\cdot\rangle_+$ the pooled mean over all other
non-padding positions. Then

$$
\text{v-ratio}=\frac{1}{L}\sum_{l=1}^{L}
\frac{\langle m^{(l)}\rangle_0}{\langle m^{(l)}\rangle_+},
\qquad
\text{h-ratio}=\frac{1}{L}\sum_{l=1}^{L}
\frac{\langle r^{(l)}\rangle_0}{\langle r^{(l)}\rangle_+}.
$$

A v-ratio below 1 indicates relative value-norm drain, and above 1 amplification.
The main v-ratio averages 30 layer ratios after pooling the three KV-group norms within
each layer. Section 4.3 separately retains all 90 group ratios. The h-ratio measures
residual-norm asymmetry and serves as a proxy for channel-level massive activations [2, 5].
Appendix E checks how position-0 measurements relate to extrema elsewhere in the sequence.
