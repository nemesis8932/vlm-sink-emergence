# Session-4 n=3 dissociation — Auditor adjudication

Auditor re-derived all seed-1/seed-2 numbers first-hand from `runs/*_seed{1,2}/probes.jsonl`.
seed-0 not re-derivable (raw absent locally; preserved in `archive/session3/gate_summary.txt`,
sha256 in archive) — trusted, flagged. Verdict adjudicated Director+Auditor.

## ⚠ Data-integrity finding (kills one EM flag)

The EM consolidated table mixed **two different metrics in one `max_a0` column**:
seed-0/seed-1 rows carried *mean* attn→pos0 (from gate_summary), seed-2 rows carried *max*
attn→pos0 (from raw). Re-derived like-for-like:

| g1gate | mean_attn_pos0 | max_attn_pos0 |
|---|---|---|
| s1 | 0.073 | 0.211 |
| s2 | 0.072 | 0.225 |

→ **The "s2 max_a0 = 0.22 anomaly / borderline head unique to s2" is an artifact.** s1 and s2
are identical on both metrics. Both seeds reproducibly show one head at ~0.21 max while mean
stays ~0.07 and the sink-fraction stays ~0. **No corner break, no s2 anomaly — g1gate is more
reproducible than the n=2 read suggested.** Same mislabel ran through the textinit rows.

## Corrected n=3 table (consistent metrics; mean attn→pos0)

Matched ~100M tok; textinit at its 60M 3-way floor (seed-0 stopped there). baseline/sigmoid n=2.

| arm | seed | tok | Sink^0.2_1 | mean_a0 | v_ratio | h_ratio |
|---|---|---|---|---|---|---|
| baseline | s0 | 101M | 0.000 | 0.068 | 0.723 | 2.16 |
| baseline | s1 | 100M | 0.000 | 0.062 | 0.687 | 1.71 |
| g1gate | s0 | 101M | 0.004 | 0.068 | 0.805 | 1.67 |
| g1gate | s1 | 100M | 0.011 | 0.073 | 0.845 | 2.04 |
| g1gate | s2 | 100M | 0.0037 | 0.072 | 0.854 | 2.24 |
| sigmoid | s0 | 102M | 0.830 | 0.377 | 1.479 | 1.10 |
| sigmoid | s1 | 100M | 0.756 | 0.311 | 1.603 | 1.30 |
| textinit | s0 | 60M | 0.852 | 0.627 | 0.377 | 42.5 |
| textinit | s1 | 60M | 0.556 | 0.235 | 0.634 | 5.50 |
| textinit | s2 | 60M | 0.578 | 0.232 | 0.484 | 12.2 |

(max_attn_pos0 available for s1/s2 only — g1gate ~0.21–0.22, textinit ~0.53–0.59 — not s0,
so kept out of the comparative column.)

## Four distinct corners — CONFIRMED (no two arms share a signature triple)

| arm | concentration | value-norm | massive-activation |
|---|---|---|---|
| baseline | none (0.00) | mild drain (<1) | moderate (h~1.7–2.2) |
| g1gate | suppressed (~0.004–0.011) | **none** (~1, 0.81–0.85) | moderate (h~1.7–2.2) |
| sigmoid | strong (0.76–0.83) | **amplified** (>1, 1.48–1.60) | none (h~1.1–1.3) |
| textinit | total (0.56–0.85) | strong drain (0.38–0.63) | extreme (h 5.5–42.5) |

value-norm alone takes three directions (drain / neutral / amplify) → signatures genuinely
decouple. g1gate vs baseline separate on value-norm; sigmoid vs textinit separate on
value-norm direction + massive-activation.

## Per-arm reproducibility read (for the verdict)

- **baseline** (n=2) — tight (sink 0/0); no-sink corner solid.
- **g1gate** (n=3) — **reproducible.** Sink^0.2 = 0.004 / 0.011 / 0.0037, all ≪0.05; mean_a0
  and max_a0 both flat across seeds. EM's "max_a0 anomaly" withdrawn (metric mislabel). Tightest arm.
- **sigmoid** (n=2) — tight; strong-concentration / value-amplified / no-MA corner solid & distinct.
- **textinit** (n=3) — **corner robust, magnitude NOT reproducible.** All 3 seeds = high
  concentration + value-drain + h_ratio>5. But seed-0 is the consistent high outlier across
  *every* signature (sink 0.85 vs 0.56–0.58; mean_a0 0.63 vs 0.23; h_ratio 42.5 vs 5.5–12.2);
  s1≈s2. h_ratio has plateaued by 60–100M (s1 5.5→4.5, s2 12.2→11.5), so the spread is a genuine
  seed effect, not an un-converged transient. **Report h_ratio as median ~12× / range 5.5–42.5,
  not a point.** Same caveat applies to textinit concentration/value magnitudes.

## Verdict line

**Four-way dissociation CONFIRMED at n=3. Four corners distinct and reproducible in kind.**
g1gate fully reproducible (concentration suppression robust; EM max_a0 anomaly was a metric
mislabel — withdrawn). textinit corner qualitatively robust but magnitude seed-sensitive
(seed-0 outlier across all signatures) → report textinit signatures as range/median, not point.
seed-0 numbers trusted from archive, not first-hand re-derivable.
