TO: Engineering Manager   FROM: Director   RE: Session-4 (close Gate B at n=3) on the CURRENT box, then exit — stop only

CONTEXT (rest in repo):
- Status: Gate A CLOSED (RF CONFIRM). Gate B OPEN at n=2. Per `archive/session3/gate_summary.txt`, only TWO arms lack n=3: g1gate (fragile, sink^0.2 0.004/0.011) and textinit (h_ratio DISAGREES 42.5 vs 5.5). baseline + sigmoid are solid at n=2 — do NOT rerun them.
- These reruns are REPEATED-data / map-style (the_cauldron 4 subsets) — no streaming, NO leak, NO litdata. Run on the current box as-is, before exit.
- Stage-2 + new instance + litdata/monitoring = DEFERRED (see `handoff-em-stage2-infra.md`); not in scope here.

ASK — run on current box, before teardown:

1. **Session-4 seed-3 runs** (cloud-agent). Third seed (`--seed 2`) of:
   - `--arm g1gate`  → match seed-0/1 budget (~100M tok). Signal: **Sink^0.2_1** (concentration) — is the near-zero reproducible at n=3?
   - `--arm textinit` → match seed-0/1 budget (~60M tok). Signal: **h_ratio** (massive-activation) — does seed-3 converge toward 42 or 5.5? (concentration logged free.)
   Recipe + probe schedule **byte-identical to the existing seed-0/1 runs** (repeated data, pretrained ViT, same LRs/wd/probe_every) — the only difference is the seed. Commit `probes.jsonl`/`train_log.jsonl`/`run_config.json` to git + HF after each.

2. **Auditor adjudication** (route to Auditor). Compact the n=3 dissociation (all 3 seeds × 4 arms) into the decoupling table; flag whether g1gate concentration is reproducible and whether textinit h_ratio converges. **Verdict is adjudicated by Director + Auditor** — Director's reference rule (g1gate Sink^0.2<~0.05/seed; textinit 2-of-3 within ~2×; 4 distinct corners) is guidance, NOT an auto-gate. Produce the publication n=3 figure.

3. **Exit** — only after Session-4 data is committed + Auditor has the n=3 result. Follow `docs/exit-runbook.md`: build generate.py → quality-check all arms → save_pretrained keepers to HF → box-only sweep → verify load elsewhere → `vastai stop` → confirm `actual_status=stopped`. **STOP THERE — agents never destroy; the user destroys the box manually.**

CONSTRAINTS: vast.ai billed; self-stop on waste; HF storage finite. Director holds go/no-go; destroy is user-only.

RETURN (compacted, no raw logs):
- Session-4: g1gate seed-2 Sink^0.2_1 + textinit seed-2 h_ratio (+ concentration), vs the existing two seeds.
- Auditor n=3 dissociation table + figure + reproducibility/convergence read for the Director+Auditor verdict.
- Exit: arm quality-check md; load-verified-elsewhere confirmation; stop confirmed (`actual_status=stopped`); final compute log. Hand back to user for destroy.
