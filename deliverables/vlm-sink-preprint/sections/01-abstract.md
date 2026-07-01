<div class="abstract">
<p><span class="abstract-head">Abstract.</span>
Attention concentration on early tokens, near-zero value-vector norms at the attended token,
and massive residual-stream activations co-occur so reliably in trained language models that
they are often treated, implicitly, as facets of a single &ldquo;attention-sink&rdquo;
phenomenon. We test that reading in the one setting where the signatures can be watched
forming rather than inferred post hoc: from-scratch multimodal pretraining. We train a
222M-parameter vision&ndash;language model (a SigLIP-B/16 encoder feeding a randomly
initialized SmolLM2-135M-architecture decoder) under four training levers &mdash; standard
softmax attention, output-gated softmax, unnormalized sigmoid attention, and decoder
initialization from a pretrained text LM &mdash; and log the three signatures as separate
per-head quantities throughout training. The three signatures come apart. Across three
seeds, the four levers land in four distinct corners of (concentration &times; value-norm
&times; massive-activation) space: no two arms share a signature triple, and the value-norm
ratio alone is drained (0.38&ndash;0.72), unchanged (&asymp;1), or amplified
(1.48&ndash;1.60) depending on the lever. On a confound-free run over one billion
non-repeated tokens, massive activation grows +130% while attention concentration remains
exactly zero for the entire run. Per-head correlation between concentration and value-norm
flips sign across arms (+0.67 to &minus;0.76; pooled &minus;0.20), inconsistent with a
single mechanism driving both. Prior text-only work separates massive activations from
attention sinks; we show, for the first time in from-scratch multimodal pretraining, that
all three signatures &mdash; including value-norm drain as a third, independently moving
axis &mdash; are separately controllable by ordinary training-time choices.</p>
</div>
