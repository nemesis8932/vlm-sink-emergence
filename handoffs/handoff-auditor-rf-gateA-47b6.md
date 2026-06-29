TO: Auditor   FROM: Engineering Manager   RE: audit RF Gate-A CONFIRM — rule the cleanliness gaps in or out before it goes publication-grade

CONTEXT (rest in repo):
- RF fresh baseline ran to 1B tok (step 69775, tok 1,001,238,892). Headline verdict: concentration sink ABSENT, decoupling confirmed, free of the repeated-data confound. Strong, not a close call (Sink^0.3_1 = 0.000 the whole run).
- Source data: git `80f55e7` → `runs/rf_fresh_baseline/` (`GATE_A_REPORT.md`, `probes.jsonl`, `train_log.jsonl`, `stdout_full.log` = authoritative, `run_config.json`, `*.pre_resume4000`). Ckpts on HF `nemesismaniac/vlm-sink-emergence-ckpts/runs/rf_fresh_baseline/` (9, step 0…64000).
- Decision frame: ADR-0001 (fresh accepts domain-shift to kill repetition), ADR-0002 (recipe locked = Comparator clone). Comparator recipe in memory `project-comparator-recipe`.

ASK: the verdict is robust (0.000 sink); your job is whether the gaps BIAS the *decoupling story* or weaken the publication claim — rule each in/out, don't re-litigate the headline:
1. **resume@4000 + optimizer reset** — Adam state reset mid-run. Comparator was one clean cosine. Does the reset perturb the 0–57M trajectory or the v_ratio/h_ratio signatures? Stitch the `*.pre_resume4000` segment to the main trajectory and check for a discontinuity at step 4000.
2. **shuffle buf=500** (built loader specced buf=10_000) — does the weaker shuffle introduce local data correlation that could flatter "no sink"? Assess; flag if the deviation needs a rerun.
3. **2.39 effective epochs** — RF was meant low-repeat. Quantify residual repetition risk vs the Comparator's ~74 epochs; is 2.39× still "effectively fresh" for the claim?
4. trajectory split + 915M-not-exact-1B + the data-continuity hygiene — confirm cosmetic, not substantive.

CONSTRAINTS: read-only on results; no GPU (box stopped). If a gap is verdict-threatening, say so plainly + name the cheapest fix (rerun param vs caveat-in-REPORT).

RETURN: per-gap in/out call; one publication-grade decoupling figure (v_ratio↓ + h_ratio↑ + sink-flat, 0→1B); a one-line "CONFIRM stands / CONFIRM with caveats / needs rerun" to the Director.
