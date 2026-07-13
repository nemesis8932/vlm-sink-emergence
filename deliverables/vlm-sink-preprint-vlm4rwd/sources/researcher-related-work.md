# Literature Task for the nanoVLM-222M Attention-Sink Paper: Wording-Lock (ASK A) + Scoop Recheck (ASK B)

## TL;DR
- **ASK A (wording-lock):** Cite the coupling literature as describing *co-occurrence / co-emergence / interacting mechanisms* in **text** LMs — not as "one inseparable phenomenon." Gu et al. (2410.10781) explicitly tie the sink to *small value-vector norms* ("non-informative and not contribute to the value computation") and link it to massive activations; Queipo-de-Llano et al. (2510.06477) are the one group that genuinely asserts **causal unity** (massive activations *drive* both attention sinks and compression valleys), so quote them precisely. Sun, Canziani, LeCun & Zhu (2603.05498) already **decouple massive activations from attention sinks in text models**, so your paper must claim novelty on *multimodal + from-scratch training-dynamics + a three-way* decoupling, not on the decoupling concept itself.
- **ASK B (scoop recheck): CLEAR.** As of June 29, 2026 no paper does from-scratch **multimodal** sink-emergence tracked across pretraining checkpoints, and none does a genuine **three-way** (attention-concentration vs value-norm drain vs massive-activation) decoupling. The nearest works are all (i) text-only two-way decouplings, (ii) text-only from-scratch emergence, or (iii) frozen-backbone / inference-time multimodal analyses.
- **Action:** Adopt the precise safe wording below; in related work cite-and-distinguish 2603.05498 and 2603.17771 (closest decoupling precedents, both text-only/two-way) and 2510.08510 / 2604.03316 (multimodal but frozen/inference-time).

---

## Key Findings

### ASK A — exactly what the coupling papers claim

**1. Gu et al., arXiv:2410.10781 — "When Attention Sink Emerges in Language Models: An Empirical View" (ICLR 2025, Spotlight).**

*Value behavior at the sink token.* Their central characterization, repeated verbatim in the abstract and in §7.3 (attention biases), is:
> "attention sink acts more like key biases, storing extra attention scores, which could be non-informative and not contribute to the value computation."

They support this geometrically: "the ℓ2-norm of keys and values of the first token is significantly smaller than that of other tokens" (also observed in Devoto et al. 2024; Guo et al. 2024b). In their K-bias intervention they set the value bias v* = 0 and find sinks still form on the key bias, "reaffirm[ing] that the sink token acts as key biases … and not contribut[ing] to the value computation." So Gu et al. **do** assert the sink token has small value norms / does not contribute to value computation — this is the value-norm-drain symptom, treated as part of the *same* sink phenomenon.

*Connection to massive activations.* They note Cancedda (2024) "attributed the phenomenon to the large norm of hidden states of the first token. This is referred to as massive activations … in Sun et al. (2024)," and they observe "massive activations on the first token appear" in standard configurations; crucially, their no-sink configurations also have **no** massive activations.

*Mechanism.* The sink "(at least partially) stems from tokens' inner dependence on attention scores as a result of softmax normalization." Relaxing this — "replacing softmax attention with other attention operations, such as sigmoid attention without normalization" — means "attention sinks do not emerge in LMs up to 1B parameters." Quantitatively, their 1B sigmoid-attention (no normalization) model reaches a validation loss of 3.10 versus 3.07 for the softmax baseline, with the sink metric "significantly" reduced — i.e., the no-softmax lever removes the sink at near-zero quality cost. (This directly grounds your sigmoid/no-softmax arm.)

> **Takeaway for your paper:** Gu et al. treat attention concentration, near-zero sink value norms, and massive activations as **co-emerging facets of one softmax-induced phenomenon** — they do not separate them. This is exactly the coupling your paper breaks apart, and Gu et al. is the correct primary citation for "value-state drain is treated as part of the sink."

**2. Queipo-de-Llano, Arroyo, Barbero, Dong, Bronstein, LeCun & Shwartz-Ziv, arXiv:2510.06477 — "Attention Sinks and Compression Valleys in LLMs are Two Sides of the Same Coin."**

This is the **one paper that asserts genuine causal unity** (not mere correlation), so cite it carefully:
> "We reveal that attention sinks and compression valleys are two manifestations of a single mechanism: massive activations in the residual stream."

They go beyond co-occurrence to causation: they "prove that massive activations mathematically require compression" (Theorem 1, spectral dominance) and "validate causality through targeted ablations: removing massive activations eliminates both compression and reduces attention sinks." Their causal test is concrete — ablating LLaMA3-8B's layer-0 massive activation "eliminates sink formation."

*Emergence dynamics (relevant to your training-dynamics framing).* Across training checkpoints (steps 1, 1k, 2k, 4k, 8k, 10k, 20k, 30k, 143k for Pythia 410M/6.9B/12B), "all three phenomena emerge together around step 1k and remain synchronized throughout training, indicating this organization is learned early." The alignment is tight: "We compute the Pearson correlation between the change in BOS norm and entropy, obtaining **r = −0.9 ± 0.18** across models, while BOS norm and sink rate correlate at **r = 0.58 ± 0.25**." When the BOS norm "spikes to factors of **10³–10⁴** (typically layers 0–5 depending on model depth), entropy simultaneously drops [below 0.5 bits] and sink rates surge [to near 1.0]"; in Pythia 410M "the transition consistently occurs at layer 5 regardless of input."

*Scope (important for honest positioning).* This is **text-only**, validated across Pythia 410M/6.9B/12B, LLaMA3 8B/70B, Qwen2 7B/72B, Gemma 7B, Bloom 1.7B and GPT-OSS 120B (the "410M–120B" range), on GSM8K. Their third axis is the **compression valley / representational entropy**, *not* value-norm drain. So their "two sides of the same coin" unifies {sinks, compression valleys} via massive activations — it does not claim your specific three (concentration / value-drain / massive activation) are one thing.

**3. Sun, Canziani, LeCun & Zhu, arXiv:2603.05498 — "The Spike, the Sparse and the Sink: Anatomy of Massive Activations and Attention Sinks."**

This paper **already decouples massive activations from attention sinks in text models**, so your paper must NOT claim the decoupling concept itself as novel:
> "the co-occurrence is largely an architectural artifact of modern Transformer design, and that the two phenomena serve related but distinct functions."

*Lever and quantified evidence.* Normalization is their decoupling lever. Their Table 5 shows Sandwich normalization yields a 44.7% sink ratio (spike value 520) versus the Pre-Norm baseline's 46.0% (spike 3818); DynamicTanh gives the highest 61.0% sink ratio (spike 153); QKNorm gives 42.0% (spike 92) — i.e., spikes are crushed while the sink ratio survives, "demonstrating that sinks can exist independently of massive activations." Their conclusion: the frequent co-occurrence "reflects incidental architectural interactions rather than a deep functional coupling."

> **Takeaway:** This is a **two-way** (massive-activation vs attention-sink) decoupling, **text-only**, via **pre-norm/normalization** levers — distinct from your **three-way**, **multimodal**, **from-scratch**, 4-lever (softmax baseline / sigmoid no-softmax / Qiu G1 gating / text-vs-random-init decoder) decoupling. Cite it as the closest prior decoupling and frame your contribution as (a) adding value-norm drain as an independent third axis, (b) multimodal training, and (c) emergence dynamics from random init rather than post-hoc analysis of trained checkpoints.

### The precise safe wording (what the paper can write)

**DEFENSIBLE — co-occurrence/co-emergence (true, citeable):**
> "In text language models, attention concentration on the first token, near-zero value-vector norms at the sink ('value-state drain'), and massive residual-stream activations co-occur and have been traced to shared or interacting mechanisms (Gu et al., 2025; Queipo-de-Llano et al., 2025; Guo et al., 2024)."

**DEFENSIBLE — the one true causal-unity claim (attribute narrowly to 2510.06477):**
> "Queipo-de-Llano et al. (2025) go further, arguing that in text LLMs attention sinks and compression valleys are causally unified by massive activations in the residual stream."

**DEFENSIBLE — acknowledging prior decoupling so you are not scooped:**
> "Concurrent text-only work shows massive activations and attention sinks are dissociable architectural artifacts rather than a single mechanism (Sun et al., 2026), and can be separated via the value/gradient path (Chen & Yao, 2026). We extend this in three ways: to the multimodal setting, across from-scratch pretraining dynamics, and by separating a third axis — value-norm drain — from attention concentration and massive activations."

**AVOID — the strawman:**
> ✗ "The literature claims these three symptoms are one inseparable phenomenon."

This overstates the field: (i) 2603.05498 already separates two of the three; (ii) the coupling papers argue *co-emergence* or *shared cause*, not *inseparability*. Saying "treated as coupled / co-occurring and often attributed to a shared mechanism" is accurate; saying "treated as one inseparable phenomenon" is a strawman that a reviewer who knows 2603.05498 will flag.

---

### ASK B — Scoop recheck: **CLEAR**

No paper (any date, and specifically nothing after ~June 10, 2026) combines **from-scratch multimodal sink-emergence across checkpoints** with a **three-way** (concentration / value-drain / massive-activation) decoupling. The nearest neighbors:

| arXiv ID | One-line overlap | How it differs from your claim |
|---|---|---|
| **2603.05498** (Sun, Canziani, LeCun, Zhu — "The Spike, the Sparse and the Sink") | Decouples massive activations vs attention sinks; shows they are dissociable artifacts | **Two-way** (not three; no value-drain axis), **text-only**, lever is pre-norm/normalization (not your 4 levers), not from-scratch emergence tracking |
| **2603.17771** (Chen & Yao — "Attention Sinks Induce Gradient Sinks") | From-scratch text LMs (0.1B/0.3B), dense checkpoints every 1k steps; V-scale intervention **preserves sinks while suppressing massive activations** | **Two-way** via gradient/value path, **text-only**, no value-norm-drain as separate signature, not multimodal |
| **2604.03316** (Choi et al. — "When Sinks Help or Hurt", LVLM) | Multimodal; distinguishes ViT-emerged (V-sink) vs LLM-emerged (L-sink); Layer-wise Sink Gating | **Frozen LVLM backbone, inference-time**; no from-scratch emergence; no three-signature decoupling |
| **2510.08510** (Luo et al. — "To Sink or Not to Sink", LVLM) | Multimodal; identifies high-norm ViT attention sinks; DIYSink module | Inference/representation focus on a trained model; not from-scratch sink-signature emergence across pretraining |
| **2510.06477** (Queipo-de-Llano et al. — "Compression Valleys") | Text-only causal unification of sinks + compression via massive activations | Opposite framing (unifies, not decouples); text-only; third axis is compression valley, not value-drain |

**At-risk sentence check.** The sentence most exposed to a scoop challenge is any claim of the form *"we are the first to decouple attention concentration, value-norm drain, and massive activations."* Because 2603.05498 and 2603.17771 already separate **two** of these in text models, do **not** claim first-ever decoupling unqualified. **Cheapest pivot (no experiments needed):** scope the novelty to the conjunction — *"first to show all three sink signatures are independently controllable during from-scratch **multimodal** pretraining, separating value-norm drain as a third axis beyond the massive-activation-vs-sink dissociation shown in text models (Sun et al., 2026; Chen & Yao, 2026)."* This preserves novelty and pre-empts the obvious reviewer objection.

---

## Details

- **Your G1 lever is well-grounded and text-only.** Qiu et al. (2505.06708, "Gated Attention for LLMs," NeurIPS 2025 Best/Oral) define G1 as a head-specific, element-wise sigmoid gate applied to the **SDPA output** (before the output projection). They report G1 "largely reduces the attention score allocated to the first token and decreases massive activations" (e.g., a layer with an 83%-on-first-token sink drops to 4%), eliminating both the sink and massive activation while improving PPL/MMLU. This is the canonical citation for your "Qiu-style G1 output gating" arm; note it is text/LLM-MoE only, reinforcing your multimodal novelty.
- **The unifying counter-view exists.** Qiu et al. (2601.22966, "A Unified View of Attention and Residual Sinks: Outlier-Driven Rescaling is Essential for Transformer Training") argue outliers (attention sinks + residual sinks) jointly with normalization perform an *essential* rescaling — a coupling/unification stance, opposite to decoupling. Cite it to show the field is actively debating coupling vs separability (text-only).
- **From-scratch text emergence precedent.** Peng et al. (2603.06591, "How Attention Sinks Emerge … An Interpretability Perspective") trace a "P0-Sink Circuit" that emerges early in a from-scratch 30B-A3B MoE and concentrates in the first two layers — text-only, no multimodal, no three-way decoupling; good support for "sinks emerge early in pretraining."
- **Active–dormant coupling of sink↔value-drain.** Guo et al. (2024, active-dormant attention heads) is the foundational citation for the *coupling* of attention sinks and value-state drains ("dormant head" = sink value state actively drained to near-zero). This is the specific coupling your value-norm-drain axis breaks; cite it alongside Gu et al.
- **Broader survey to cite.** Su et al. (2604.10098, "Attention Sink in Transformers: A Survey," v2 June 5 2026) organizes the field into Utilization / Interpretation / Mitigation — a convenient single citation for "attention sinks are pervasive across modalities."

---

## Recommendations

1. **Lock the safe wording (do now).** Use the "co-occur / interacting mechanisms" phrasing for Gu et al. + Guo et al.; reserve "causally unified" strictly for Queipo-de-Llano et al. (2510.06477); never write "one inseparable phenomenon."
2. **Add a 3–4 sentence related-work paragraph** that explicitly cite-and-distinguishes 2603.05498 (two-way, text, pre-norm lever) and 2603.17771 (two-way, text, value/gradient path, from-scratch w/ checkpoints), then states your three-axis + multimodal + from-scratch novelty. This is the single most important edit to survive review.
3. **Scope the novelty claim to the conjunction** (multimodal × from-scratch emergence × three-way), per the pivot above. Benchmark that would change this recommendation: if a reviewer surfaces a paper that separates value-norm drain as an independent third axis in *any* setting, downgrade to "first in the multimodal/from-scratch setting."
4. **Do one final manual browse of arxiv.org/list/cs.CV/2606 and cs.LG/2606 (June 10–29)** before posting — sub-two-week preprints may not yet be search-indexed; this is the only residual risk.
5. **Cite Qiu 2505.06708 for G1 and Gu 2410.10781 for sigmoid/no-softmax** as the established (text-only) precedents for two of your four levers; explicitly note the text-init-vs-random-init-decoder lever has no sink-context precedent (novel control).

---

## Caveats
- arXiv IDs with 26xx month-codes correspond to 2026 months (e.g., 2603 = March 2026); submission dates on the abstract pages were treated as authoritative. IDs 2410.10781, 2510.06477, 2603.05498, 2603.17771, 2604.03316, 2510.08510, 2505.06708 and 2601.22966 were each verified against their arXiv abstract/HTML pages.
- The numbers for 2510.06477 (r = −0.9 ± 0.18; r = 0.58 ± 0.25; 10³–10⁴ BOS-norm spike; emergence ~step 1k) and for 2603.05498 (Table 5 sink-ratio/spike values) are quoted from the papers' own text/tables.
- "CLEAR" reflects no scooping work found across arXiv/Scholar/OpenReview searches; absence of a very recent (<2-week) preprint cannot be fully excluded, hence Recommendation 4. No "nanoVLM attention sink" competitor was found.
- 2510.06477's lead authorship is listed as Queipo-de-Llano and Arroyo (equal first authorship); cite accordingly rather than as "Queipo-de-Llano et al." alone if your venue requires both.