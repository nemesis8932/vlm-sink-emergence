# 2. Related work

**Coupled signatures in text models.** Gu et al. [6] connect attention concentration to
small value norms and large residual activations, interpreting the sink as a key bias.
Guo et al. [4] study the associated value-drain mechanism through active-dormant attention
heads. Queipo-de-Llano et al. [7] link massive activations to compression valleys, layers
where token representations become less diverse. Their targeted activation ablations
also suppress sink formation in the studied text models. Qiu et al. [8] explain attention
and residual sinks through outlier-driven rescaling, while Peng et al. [24] trace a
position-zero sink circuit during text pretraining. These accounts motivate measuring the
signatures together without assuming their relationship transfers unchanged to VLMs.

**Dissociations and interventions.** Sun et al. [9] preserve attention sinks while
suppressing massive activations by changing normalization. Chen, Lin and Yao [10] obtain
a related dissociation with V-scale, which modifies gradients on the value path. Our study
adds joint tracking of value-norm drain during multimodal pretraining. Fesser et al. [23]
further distinguish a negligible-value sink that suppresses a head's update from a
broadcast sink that redistributes information in trained vision transformers. Their
analysis supports separating attention allocation from the value vectors it weights.
For interventions, we adapt the unnormalized sigmoid attention studied by Gu et al. [6]
and the attention-output gate of Qiu et al. [20]. Our gate's initialization and associated
scale change are specified in Section 3.1.

**Multimodal sinks and grounding.** Luo et al. [11] distinguish vision-encoder-propagated
and decoder-emerged sinks. Their alignment-checkpoint analysis provides training-time
evidence for activation growth, using a frozen vision encoder and a pretrained language
decoder. Choi et al. [12] distinguish vision and language sinks and train layer-wise sink
gates while keeping the underlying VLM frozen. We study a different regime, jointly
tracking attention, value norms and residual norms during pretraining with random decoder
initialization. High-norm tokens can also arise inside pretrained vision transformers
[25], which matters because our encoder is pretrained. Work on sink-aware visual attention
and decoding connects these internal patterns to grounding and hallucination [13–15].
Sink statistics have also been studied as hallucination signals in text LMs [16]. Our
experiments concern the training dynamics of the signatures rather than their downstream
effects.
