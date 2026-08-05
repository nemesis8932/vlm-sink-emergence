# Open questions & reviewer-critique pre-empts

Surfaced in Director grill (2026-06-15). These are the known soft spots in the
decoupling result + the cheap checks that close them. Pick up before the writeup.
Owners: A=Auditor, EM=Engineering Manager, D=Director.

## 1. Are we only looking at pos0? (per-position robustness) — A
The headline `Sink^ε` metric and the v_ratio/h_ratio ratios are **pos0-anchored**. A
sink parked on a *different* image token, or a massive-activation on a non-pos0 token,
would be diluted into "rest" and partly missed. We do log per-head argmax + segment
mass (Session-1: 80% argmax at pos0, none in text), but no per-position norm scan.
**Cheap fix, zero compute:** from existing `reprobe/` per-head/per-position detail
(Mac, off-cloud) produce (a) argmax-position histogram across all positions, (b)
per-position v-norm / h-norm profiles → prove pos0 is *the* special token, not assumed.
Run on RF + Session-3 reprobe data once RF reprobe lands.

## 2. Pretrained ViT = partial "from scratch" + inherited-norm confound — A/EM
RF decoder is random but the SigLIP vision encoder is **pretrained**. ViTs grow
high-norm register/artifact tokens (Darcet et al., *ViTs Need Registers*); 2510.08510
separates ViT-propagated sinks from LLM-emerged ones. So our pos0 massive-activation
*could* be inherited from SigLIP, not decoder-formed.
**Defense we already have:** h_ratio starts ~1.0–1.4 at step 0 and *rises* during
training → forming, not inherited (pure inheritance would be high at init).
**Disentangler (follow-up):** `--vit_init random` variant + probe ViT's own attention.
List pretrained-ViT as an explicit limitation; cite register-token / propagated-sink work.

## 3. "Symptoms assumed inseparable" — pin the wording — D
We have evidence symptoms **co-occur** in text LLMs (Gu et al. 2410.10781;
2510.06477 co-emergence ~step 1k) and **don't** in VLMs (ours). "Assumed inseparable"
is rhetorical sharpening — defensible only if we cite the coupling papers; do NOT
overstate as a formal claim they made (strawman risk). Lock exact phrasing pre-writeup.

## 4. R2 / overfitting-independence justification — D (parked)
R2 (seed-2 Gate-B) numbers were independent of the overfitting incident, but the
justification is thin and will draw critique. Triage for later; not blocking RF.

## 5. Domain-shift confound (cauldron→FineVision) — D
Accepted + documented (ADR-0001). The fresh-repeated control arm (FineVision capped to
~146K imgs, matched repetition) is the known follow-up if a reviewer presses.

## 6. Token scale vs Gu et al — D
RF = 1B tok; Gu canonical = 5B. ~5× short. Defensible (emergence is early; 1B is past
where text-LM sinks form), but a reviewer will ask. Gated Stage-2 (~3–5B/arm) + Stage-3
A100 (~5B) is the answer if results justify spend.

## 7. Single seed for RF — D
RF is one seed. Concentration=0 was rock-solid across both seeds on repeated data, so
likely sufficient for the negative claim. Add RF seed-2 only if 1B verdict is borderline.
