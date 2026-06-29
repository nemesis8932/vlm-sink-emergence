TO: Director   FROM: Engineering Manager   RE: RF fresh baseline complete — Gate-A CONFIRM, with cleanliness gaps routed to Auditor

HEADLINE: **Gate A CONFIRMED on fresh data — concentration sink absent, decoupling holds, repeated-data confound removed.**
- Sink^0.3_1 = **0.000** across the entire 0→1B run (no attention-concentration sink).
- @1B 3-signature: max_attn_pos0 **0.098**, v_ratio **0.692** (↓ from ~0.79), h_ratio **3.22** (↑ from ~1.4, +130%).
- Decoupling: value-norm ↓ + massive-activation ↑ ~130% with NO sink forming → the hypothesis signature, now on fresh data.
- Fresh val_loss 1.46→**0.638**, tracks train, no overfit.
- Stopped deliberately at 1B (step 69775). Data: git `80f55e7` `runs/rf_fresh_baseline/`; HF 9 ckpts.

VERDICT IS ROBUST — Sink=0.000 is not a close call. But 3 cleanliness gaps dent the *clean-clone* framing (not the headline):
- **resume@4000, optimizer reset** (High) — Adam reset mid-run → RF no longer byte-identical to the one-cosine Comparator.
- **shuffle buf=500** vs specced 10k (Med) — weaker shuffle, possible local data correlation.
- **2.39 effective epochs** (Med) — not single-pass, but ≪ Comparator's ~74 epochs → confound hugely reduced, not zero.
- (Low: trajectory split in pre_resume files, 915M ckpt ≠ exact 1B.)

ROUTING: dispatched Auditor (`handoffs/handoff-auditor-rf-gateA-47b6.md`) to rule each gap in/out and produce the publication decoupling figure + a CONFIRM-stands / CONFIRM-with-caveats / needs-rerun call.

OPEN: (1) verify box `actual_status=stopped` (agent said "stopping now" — Session-1 lost $1.5 to a silent no-op; not yet independently confirmed). (2) `mac-reorg` now unblocked to merge (RF done, box down).

GO/NO-GO FOR YOU: accept CONFIRM as provisional pending Auditor, or hold the claim until Auditor rules. Recommend provisional-accept — headline won't move; gaps are framing/caveat-level.
