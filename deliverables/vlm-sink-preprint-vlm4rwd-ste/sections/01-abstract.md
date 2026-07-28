<div class="abstract">
<p><span class="abstract-head">Abstract.</span>
Trained language models show three effects at the same early token. Attention
concentrates on the token. The value vector at that token has a very small norm. The
residual stream at that token has a very large norm. These three effects occur together so
often that the literature treats them as one &ldquo;attention sink&rdquo; phenomenon. We
test that reading in a setting where a reader can watch the three effects form. That
setting is from-scratch multimodal pretraining. We train a 222M-parameter
vision&ndash;language model. A SigLIP-B/16 encoder feeds a randomly initialized decoder
with the SmolLM2-135M architecture. We train this model under four levers: standard
softmax attention, output-gated softmax, unnormalized sigmoid attention, and decoder
initialization from a pretrained text language model. We log all three signatures as
separate per-head quantities through training.
The three signatures come apart. Across three seeds, the four levers reach four different
corners of (concentration &times; value-norm &times; massive-activation) space. No two arms
share a signature triple. The value-norm ratio alone moves in three directions. The lever
drains it (0.38&ndash;0.72), leaves it near 1, or amplifies it (1.48&ndash;1.60). We then
train one repetition-confound-free run over one billion fresh tokens (single seed). In that
run massive activation grows by 130 percent while attention concentration stays at exactly
zero. The per-head correlation between concentration and value-norm also flips sign across
arms (+0.67 to &minus;0.76, pooled &minus;0.20). No fixed-sign coupling links the two axes.
Earlier text-only work separates massive activations from attention sinks. We show, for the
first time in from-scratch multimodal pretraining, that all three signatures respond
independently to ordinary training-time choices. Value-norm drain is one of the three.
This result speaks to grounding as a precondition question. Attention mass that a
positional artifact captures is attention that the model does not spend on image evidence.
Other work links sink-like attention to hallucination in deployed vision&ndash;language
models. We do not test whether signature dissociation predicts grounding behavior. We
establish when and how each signature forms, which is the step that comes first.</p>
</div>
