<div class="abstract">
<p><span class="abstract-head">Abstract.</span>
Attention concentration on early tokens, near-zero value-vector norms at the attended
token, and massive residual-stream activations co-occur so reliably in trained language
models that they are often treated as facets of a single &ldquo;attention-sink&rdquo;
phenomenon. We test that reading in a setting where the signatures can be watched forming:
from-scratch multimodal pretraining. We train a 222M-parameter vision&ndash;language model
(a SigLIP-B/16 encoder feeding a randomly initialized SmolLM2-135M-architecture decoder)
under four training levers &mdash; standard softmax attention, output-gated softmax,
unnormalized sigmoid attention, and decoder initialization from a pretrained text LM
&mdash; and log all three signatures as separate per-head quantities throughout training.
They come apart. Across three seeds, the four levers land in four distinct corners of
(concentration &times; value-norm &times; massive-activation) space; no two arms share a
signature triple, and the value-norm ratio alone is drained (0.38&ndash;0.72), unchanged
(&asymp;1), or amplified (1.48&ndash;1.60) depending on the lever. On a
repetition-confound-free run over one billion fresh tokens (single seed), massive
activation grows +130% while attention concentration stays at exactly zero. Per-head
correlation between concentration and value-norm flips sign across arms (+0.67 to
&minus;0.76; pooled &minus;0.20): no fixed-sign coupling links the two axes. Prior
text-only work separates massive activations from
attention sinks; we show, for the first time in from-scratch multimodal pretraining, that
all three signatures, value-norm drain included, respond independently to ordinary
training-time choices. For grounded generation this is a precondition question: attention
mass captured by a positional artifact is, by construction, attention not spent on image
evidence, and sink-like attention has repeatedly been linked to hallucination in deployed
VLMs. Whether signature dissociation predicts grounding behavior is not tested here; we
establish when and how each signature forms, the step that comes first.</p>
</div>
