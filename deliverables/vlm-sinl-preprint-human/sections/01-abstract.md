## Abstract

Hallucination is among the most costly failure modes of vision–language models, and
attention sinks — positions that absorb disproportionate attention regardless of content —
are among the signals most often implicated. In trained language models, attention
concentration on early tokens, near-zero value-vector norms at the attended token, and
massive residual-stream activations co-occur so reliably that they are often treated as
facets of a single "attention-sink" phenomenon. We test that reading in a setting where the
signatures can be watched forming: from-scratch multimodal pretraining. We train a
222M-parameter vision–language model (a SigLIP-B/16 encoder feeding a randomly initialized
SmolLM2-135M-architecture decoder, where position 0 is the first image token and there is no
BOS) under four training levers — standard softmax attention, output-gated softmax,
unnormalized sigmoid attention, and decoder initialization from a pretrained text LM — and
log all three signatures as separate per-head quantities densely throughout training.

They come apart. Across n = 2–3 seeds per arm, the four levers land in four distinct corners
of (concentration × value-norm × massive-activation) space; no two arms share a signature
triple, and the value-norm ratio alone is drained (0.38–0.72), unchanged (≈1), or amplified
(1.48–1.60) depending on the lever. On a repetition-confound-free run over one billion fresh
tokens (single seed), massive activation grows +130% while attention concentration stays at
exactly zero. Per-head correlation between concentration and value-norm flips sign across
arms (+0.67 to −0.76; pooled −0.20): no fixed-sign coupling links the two axes.

Prior text-only work separates massive activations from attention sinks [3, 4]; we show, for
the first time in from-scratch multimodal pretraining, that all three signatures, value-norm
drain included, respond independently to ordinary training-time choices. Whether signature
dissociation predicts grounding or hallucination behavior is not tested here; we establish
when and how each signature forms, the step that comes first.
