# Preprint-readiness audit (Auditor → Director)

Zero-GPU, first-hand from local data. Four items requested. **Item #1 (the priority) is
NOT fully closeable from local data and what IS local raises a real concern — see ⚠ below.**
Items #2–4 delivered.

---

## ⚠ ITEM #1 — Per-position robustness: PARTIAL + flag (does NOT cleanly confirm pos0)

**Bottom line: pos0-as-THE-special-token is confirmed only for sigmoid. It is contradicted
for textinit and unprovable-from-local-data for the no-sink arms.**

### What's blocked
The per-position v-norm / h-norm *profiles* (item 1b) require `reprobe_step*.npz` (reprobe.py
product "3c"). Those are **gitignored (`*.npz`) and not present anywhere on this Mac** — they
live on HF / the cloud box only. Only `runs/rf_fresh_baseline/ckpt_step64000.pt` is local
(no other arm's checkpoint), so reprobe cannot be re-run locally either. **1b cannot be done
without fetching the npz from HF.**

### What's available + what it shows (item 1a, argmax-attention position)
Built from `argmax_key` in the live `probes.jsonl` (the one genuine per-position signal in git).
Figure: `analysis/argmax_position_by_arm.svg`. Fraction of (layer,head) whose **argmax key**
sits at each position, @~100M:

| arm | pos0 | pos1 | pos2 | pos3–9 | pos10+ | mean mass→pos0 | reading |
|---|---|---|---|---|---|---|---|
| baseline | 69% | 17% | 4% | 6% | 3% | 0.062 | diffuse (no sink); pos0 mildly preferred |
| g1gate | 79% | 18% | 1% | 2% | 1% | 0.07 | diffuse (no sink) |
| **sigmoid** | **87%** | 7% | 6% | 0 | 0 | **0.276** | **concentrated AT pos0 — anchoring correct ✓** |
| **textinit** | 35% | 34% | 1% | 15% | 14% | 0.23 | **concentrated, but position SPLIT and seed-variable ✗** |
| RF (fresh) | 20% | **76%** | 3% | 0 | 0 | 0.046 | diffuse (no sink); argmax pos1 not pos0 |

Per-seed, textinit is worse than the pooled row suggests: **seed1 = pos0 61% + pos5 29%;
seed2 = pos1 60% + pos13 29%, pos0 only 9%.** The hot token *moves across seeds.*

### Why this matters (metric contract, `docs/conventions.md`)
All three headline signatures are **pos0-anchored by construction**: Sink^ε = fraction of
heads with mean attn→**pos0** > ε; v-ratio = ‖v‖ **pos0** vs rest; h-ratio = residual norm
**pos0** vs rest. A sink/MA parked elsewhere is a **false negative** for Sink^ε and is
**diluted into "rest"** for v/h. The argmax data shows this failure mode is realized:

- **sigmoid:** sink is at pos0 (87%, mass 0.28). pos0-anchoring valid. The clean
  strong-concentration corner stands.
- **textinit:** concentration/MA exist but the dominant position is **not stably pos0** and
  *migrates by seed*. The pos0-anchored h_ratio likely **under-measures** textinit's true
  massive-activation, and the h_ratio seed-spread (42.5 / 5.5 / 12.2) may be **position
  migration mistaken for magnitude noise** — not purely a magnitude effect. This directly
  reinterprets the Session-4 finding. Cannot be confirmed/refuted without the per-position
  h-norm profile (npz).
- **no-sink arms (baseline/g1gate/RF):** mass is diffuse everywhere (≤0.07, near uniform),
  so the *absence* of concentration is robust to anchoring **in spirit** — but a hidden
  sink at pos1 (RF argmax is pos1 76%) cannot be *excluded* first-hand without the
  per-position max-attention profile. Circumstantial evidence (low max_a0 ≈0.10, img_mass
  0.83 spread over many tokens) argues no concentration anywhere, but it is not proof.

### Impact on the headlines
- **RF decoupling (Gate A):** h_ratio rises *at pos0* (1.43→3.22) — that rise is real and
  pos0-located regardless. The decoupling statement ("MA rises at pos0 while concentration
  at pos0 stays 0") holds as written. Residual risk: a hidden pos1 concentration sink would
  weaken "concentration absent." Probably benign (diffuse mass) but not locally provable.
- **Four-way dissociation:** survives as a **qualitative** 4-corner claim. sigmoid's corner
  is cleanly pos0. textinit's magnitudes get a second caveat (position, not just seed).

**Recommendation:** this is the reviewer-facing defense — do it properly. Cheapest real fix
is **pull the reprobe npz from HF (zero-GPU download) and build the true per-position v/h
profiles**, OR reframe v1 to present sink *location* as an arm-dependent finding + caveat.

### EXECUTION STATUS (Director chose: pull npz from HF) — BLOCKED here + scope finding

1. **Cannot fetch from this Auditor session.** `pip install huggingface_hub` and direct
   `curl` to the HF API are both **denied by the environment** (even with the sandbox
   disabled) — no network egress / no installs from here. The pull must run on a networked
   device (EM / cloud box / Director's machine). Routed: `handoffs/handoff-em-perposition-fetch.md`.
2. **Scope finding — the npz only half-closes #1.** Read of `reprobe.py`: the only full
   per-position array dumped is `raw_profile` (L,H,T) = per-position **attention mass**.
   Value/residual **norms are stored pos0-vs-rest only** (`vn_pos0/vn_rest/hn_pos0/hn_rest`),
   NOT per-position. So pulling the npz closes the **concentration-location** question
   definitively (is the attention sink at pos0? — the headline concern) but **NOT** the
   per-position v/h *norm* profiles (item 1b as worded). Those need a `reprobe.py` patch to
   dump per-position norms + a re-run on checkpoints (GPU). The attention-location half is
   the one that actually defends the headline, so this is an acceptable v1 close; the norm
   half can be a camera-ready addition.
3. **Drop-in ready:** `analysis/per_position_attention_from_npz.py` — run once npz are synced;
   prints per-arm verdict (is pos0 the max-mass position?) + emits json for the figure.

### RESOLVED (2026-06-29) — npz fetched, attention-half CLOSED for seed-0

HF held reprobe npz for **seed-0 only** (4 arms, no seed1/2/RF). Per-position **mass**
(`raw_profile`, mean over layers+heads) — figure `analysis/per_position_mass_seed0.svg`,
data `analysis/per_position_attention.json`:

| arm | mass@pos0 (mean) | max-head@pos0 | argmax-mass pos | verdict |
|---|---|---|---|---|
| baseline | 0.06 | 0.17 | 0 | pos0 max; diffuse (no sink) |
| g1gate | 0.07 | 0.23 | 0 | pos0 max; diffuse |
| sigmoid | 0.30 | 0.66 | 0 | pos0 max; broad raw-sigmoid mass |
| **textinit** | **0.63** | **0.99** | **0** | **pos0 max; razor spike (pos1=0.009)** |

**pos0 is THE max-mass token in every arm at seed-0 — by mass, not just argmax.** This
**validates the reported (seed-0) headline anchoring, including textinit h_ratio 42.5**
(its sink is overwhelmingly at pos0: 0.63 mean / 0.99 max-head). The earlier argmax concern
(seed-2 textinit argmax→pos1) is now a **position seed-variability** caveat, NOT a refutation
of the seed-0 result.

**Still open (smaller, reproducibility-level):** per-position MASS for seed-1/seed-2/RF — no
npz on HF. Only live-probe argmax exists there (hints textinit position migrates at seed-2).
Closing needs cloud reprobe regen (GPU) → EM. Not v1-blocking: the headline runs are seed-0
and are confirmed.

---

## ITEM #2 — Wording lock for the "single-phenomenon" claim

**Locked sentences (use verbatim; avoids the strawman):**

> In decoder-only language models, attention concentration, value-norm collapse, and massive
> activations have been observed to **emerge together** — concentrating on the same few tokens
> (Gu et al., arXiv:2410.10781) and **co-emerging early in training**, by roughly step 1k
> (arXiv:2510.06477). Their consistent co-occurrence has led these signatures to be treated,
> often implicitly, as facets of a single "attention-sink" phenomenon. We show that in
> vision-language pretraining they **dissociate**.

Rationale: claims only what the cited work *observed* (co-occurrence / co-emergence). Does NOT
assert prior authors formally claimed the signatures "inseparable" → no strawman. Drop the word
"inseparable" from REPORT/abstract; use "treated as facets of a single phenomenon."

Citations to verify bibliographically (Researcher pass — IDs from Director/open-questions):
arXiv:2410.10781 (Gu et al.); arXiv:2510.06477 (co-emergence ~step 1k); arXiv:2510.08510
(ViT-propagated vs LLM-emerged sinks); Darcet et al. *Vision Transformers Need Registers*.

---

## ITEM #3 — Limitations section (draft prose, for REPORT / preprint §Limitations)

> **Limitations.** (1) *Per-position anchoring.* Our three signatures are measured at the first
> image token (position 0). This is verified to be the concentration site for the sigmoid arm
> (87% of heads' argmax), but in the text-initialized arm the most-attended token is not stably
> position 0 and varies across seeds; pos0-anchored magnitudes for that arm should be read as
> lower bounds. A full per-position norm scan is left to a camera-ready revision.
> (2) *Pretrained vision encoder.* The SigLIP encoder is pretrained, so each arm is only
> partially "from scratch," and ViTs are known to grow high-norm register/artifact tokens
> (Darcet et al.; arXiv:2510.08510 separates ViT-propagated from LLM-emerged sinks). Our
> defense that the massive-activation is *formed, not inherited*: h_ratio starts ≈1.0–1.4 at
> step 0 and *rises* during training (pure inheritance would be high at init). A random-ViT
> variant is future work.
> (3) *Token scale.* Runs are ≤1B tokens vs the ~5B canonical for text-LM sink studies (Gu et
> al.); emergence is early, so 1B is past where text-LM sinks form, but larger-scale
> confirmation is future work.
> (4) *Text-init magnitude reproducibility.* The text-init massive-activation is seed-sensitive
> (h_ratio range 5.5–42.5 across three seeds, seed-0 an outlier on every signature); we report
> it as a range/median, and the corner is a *kind*, not a calibrated magnitude.
> (5) *Provenance.* Seed-0 raw probes are not re-derivable first-hand (trusted from the
> checksummed `archive/session3/` summary); seed-2 was run only for g1gate and text-init.
> (6) *Domain shift.* The fresh-data control shifts domain (the_cauldron→FineVision; ADR-0001),
> accepted to remove the repeated-data confound; a domain-matched fresh-repeated control is the
> known follow-up.

---

## ITEM #4 — textinit → range/median: status

- `deliverables/session4_n3_audit.md` — already range/median (h_ratio 5.5–42.5, median ~12). ✓
- `runs/.../session4_dissociation_n3.svg` — plots individual seed points, no point estimate. ✓
- **`REPORT.md` — NOT yet; carries bare "42" at lines 66, 135, 157.** Fixing in this pass.
- **`deliverables/vlm-sink-onepager.html` — bare "42" / "0.85" / "0.38" as points.** Needs the
  same fix (flagged; HTML edit pending Director go on v1 framing).
