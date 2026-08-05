## Abstract

Hallucination is one of the most pressing limitations of large language models, and in
vision–language models it is more rampant still. No clean solution has emerged, but several
measurable symptoms track it. Attention sinks are among the most studied: positions that
absorb a disproportionate share of attention regardless of content, repeatedly linked to
hallucination in deployed VLMs [21, 22], and partially mitigated with verifiable gains. In
text language models the sink is well characterised, and it carries three signatures that
co-occur so reliably they are usually treated as facets of a single phenomenon — attention
concentration (Sink^ε_1), a near-zero value-norm at the attended token (v-ratio), and a
massive residual-stream activation (h-ratio). In vision–language models the picture is far
less settled.

We study how those three signatures emerge in a 222M-parameter VLM — a SigLIP-B/16 encoder
feeding a randomly initialized SmolLM2-135M-architecture decoder, where position 0 is the
first image token and there is no BOS — trained from scratch under four levers: standard
softmax attention, output-gated softmax, unnormalized sigmoid attention, and decoder
initialization from a pretrained text LM. Logging all three separately throughout training,
we find they come apart. Across n = 2–3 seeds per arm the four levers land in four distinct
corners of (concentration × value-norm × massive-activation) space, and no two arms share a
signature triple; the value-norm axis alone is drained, left unchanged, or amplified
depending only on the lever. On a repetition-confound-free run over one billion fresh
tokens, massive activation more than doubles while attention concentration stays at exactly
zero. The per-head relationship between concentration and value-norm flips sign from one arm
to the next, so no fixed coupling links the two.

Prior text-only work separates massive activations from attention sinks [3, 4]; we show, for
the first time in from-scratch multimodal pretraining, that all three signatures — value-norm
drain included — respond independently to ordinary training-time choices. Whether that
dissociation predicts grounding or hallucination behaviour we do not test here; we establish
when and how each signature forms, which is the step that comes first.