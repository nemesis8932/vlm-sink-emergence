# 4. Results

## 4.1 Each training condition lands in a different corner of signature space

Table 1 summarizes per-seed ranges at checkpoints near 100M tokens, or 60M for textinit
(Appendix D). The clearest separation is between sigmoid and textinit: both show
strong relative concentration, but sigmoid amplifies value norms and retains an h-ratio
near 1, whereas text initialization combines value drain with a much larger h-ratio.
Figure 2 shows the corresponding trajectories.

<figure id="fig1">
<img src="figures/fig2_phase_portrait.svg" alt="Attention maximum versus value and residual norm ratios">
<figcaption><b>Figure 2: Signature trajectories during pretraining.</b>
(a) Value-norm ratio and (b) residual-norm ratio, on a logarithmic vertical scale, against
maximum attention $a_{\max}$ (Section 3.3). Open circles mark initialization and diamonds
the final recorded checkpoints: 174M tokens for baseline, 103M for g1gate, 102M for
sigmoid, 60M for textinit, and 1B for RF. All paths use seed 0. Faint points are raw probes.
Lines are smoothed (Appendix F). The dashed horizontal reference is a norm ratio of 1.</figcaption>
</figure>

**Table 1: The four corners, ranges over 2-3 seeds per condition.** Concentration uses
Sink^0.2_1. The v-ratio and h-ratio are the layer-averaged quantities defined in Section 3.3.

| arm | concentration | value-norm ratio | residual-norm ratio |
|---|---|---|---|
| baseline | absent (0.000) | drain (0.69–0.72) | 1.7–2.2 |
| g1gate | near-absent (0.004–0.011) | drain (0.81–0.85) | 1.7–2.2 |
| sigmoid | strong (0.76–0.83) | amplified (1.48–1.60) | 1.1–1.3 |
| textinit | strong (0.56–0.85) | drain (0.38–0.63) | 5.5–42.5 |

The g1gate and baseline corners overlap on residual-norm ratio and both have little
concentration. Their value-norm ranges separate, with less drain in the gated condition.
This difference belongs to the combined gate-and-initial-scale intervention in Section 3.1.
It does not isolate a benefit of learned gating. Under sigmoid, high normalized
concentration coexists with decreasing raw attention weights to position 0 (Appendix F).
Thus, the relative attention and norm signatures separate even when the absolute
attention scale changes.

The textinit corner repeats across seeds, while its h-ratio varies from
5.5 to 42.5, with median 12.2. Per-position scans also show that attention and norm extrema
can occupy different tokens (Appendix E). Figure 3 shows a first-position attention
preference already present before multimodal training in textinit, consistent with
inheritance from the text decoder. This preference is subthreshold at $\epsilon=0.3$.
Threshold-crossing order and spatial dissociations are detailed in Appendix I.
Appendix G outlines hypotheses for the corners, and Appendix H summarizes metric provenance.

<figure id="fig2">
<img src="figures/fig6_sink_stripe.svg" alt="Selected-head query-by-key attention maps">
<figcaption><b>Figure 3: First-position attention across training conditions.</b>
Selected-head query-by-key maps at early and final checkpoints, plus textinit at
initialization, all seed 0. Each map is a batch-mean attention matrix normalized by row.
The color scale saturates at 0.5. Cyan marks key position 0, dotted lines the image/text
boundary. Heads are selected separately by condition. These qualitative maps use the
batches and selection procedure documented in Appendix F.</figcaption>
</figure>

## 4.2 Residual-norm growth without concentration over 1B tokens

The repeated-data comparison shows a substantial validation gap, with loss near 0.44
on training images paired with new questions versus 1.18 on held-out images. RF tests
whether the absence of concentration persists with less image reuse. Its held-out loss
ends at 0.638 and has a negative fitted slope over the second half of training, despite
evaluation-to-evaluation fluctuations (Appendix A).

Across RF's 698 recorded probes, none of the 270 heads exceeds 0.2 mean attention to
position 0. The largest value observed is 0.151. Consequently Sink^ε_1 is exactly zero
at all probes for $\epsilon\in\{0.2,0.3,0.4\}$. Over the same run, the residual-norm ratio
rises from 1.43 to 3.22 (Table 2). The result is a growing norm asymmetry without
thresholded attention concentration on the fixed probe batch.

**Table 2: RF, one seed, initialization to 1B tokens.**

| signature | initialization | 1B tokens | change |
|---|---|---|---|
| Sink^0.3_1 | 0.000 | 0.000 | zero at all 698 probes |
| maximum attention $a_{\max}$ | 0.056 | 0.098 | remains below 0.2 |
| h-ratio | 1.43 | 3.22 | 2.25-fold increase |
| v-ratio | 1.00 | 0.69 | net drain |

The h-ratio rises from 2.40 near 57M tokens to 3.22 at 1B, so the increase continues
well beyond the initial warmup. Figure 2b shows this change at low concentration.
The value-norm ratio follows a different trajectory: about 75% of its net decline occurs
by 57M tokens, followed by partial recovery. It therefore provides evidence of persistent
drain, not a monotonic increase in drain throughout training.

## 4.3 Concentration–value-norm correlations change sign across conditions

Table 3 reports Pearson correlations between attention to position 0 and value-norm ratio
over 90 layer/KV-group pairs per condition at seed 0. Each group's attention is averaged
over its three query heads. Its value ratio is counted once (Figure 5, Appendix C).
The correlation is positive under baseline softmax and negative under text initialization.

**Table 3: Pearson r across 90 layer/KV-group pairs at the final available reprobe, seed 0.**

| baseline | g1gate | sigmoid | textinit | RF | pooled |
|---|---|---|---|---|---|
| +0.76 | +0.57 | −0.03 | −0.79 | +0.51 | −0.20 |

These are descriptive associations within trained models. Layer dependence and the
single seed per condition preclude treating the 90 observations as independent experimental
replicates. The sign change shows that greater first-position attention is associated
with larger values in one condition and smaller values in another. It does not establish
statistical independence or rule out a nonlinear or layer-dependent relationship.
The pooled value combines within-condition and between-condition variation and is not
evidence that the signatures are unrelated.
