# Appendix

## A. Supporting figures

<figure id="figA1">
<img src="figures/fig1_layerhead_grid.svg" alt="Per-(layer,head) attention to pos0 over training">
<figcaption><b>Figure A1: Where concentration lives in the network.</b> Per-(layer, head)
attention→pos0 over training (row-normalized). baseline and RF stay cold throughout;
textinit is hot from initialization; sigmoid lights up a band of mid-network layers.</figcaption>
</figure>

<figure id="figA2">
<img src="figures/fig3_perhead_scatter.svg" alt="Per-head concentration vs value-norm scatter">
<figcaption><b>Figure A2: No universal head-level coupling.</b> Per-(layer, head)
concentration vs. value-norm ratio at the final checkpoint, by arm (n = 270/arm). The
correlation sign flips by arm (+0.67 baseline … −0.76 textinit; Table 4); the pooled cloud
is weak (−0.20) only because opposite-signed arms cancel. This should not be read as an
uncorrelated cloud — most individual arms show strong |r|.</figcaption>
</figure>

<figure id="figA3">
<img src="figures/fig5_entropy.svg" alt="Attention entropy over training by arm">
<figcaption><b>Figure A3: Entropy collapse tracks concentration only.</b> Attention entropy
over training: only sigmoid and textinit collapse; baseline, g1gate, and RF stay flat —
the entropy-collapse correlate from the text-LM literature follows the concentration axis,
not the norm axes.</figcaption>
</figure>

## B. Full per-seed signature table

Reproduces Table 1 with max attention→pos0 where available. This metric was logged for
seeds 1–2 only (the seed-0 archive summary does not include it), so it is kept out of the
main comparative table.

| arm | seed | tokens | Sink^0.2 | mean attn→pos0 | max attn→pos0 | v-ratio | h-ratio |
|---|---|---|---|---|---|---|---|
| baseline | s0 | 101M | 0.000 | 0.068 | — | 0.723 | 2.16 |
| baseline | s1 | 100M | 0.000 | 0.062 | — | 0.687 | 1.71 |
| g1gate | s0 | 101M | 0.004 | 0.068 | — | 0.805 | 1.67 |
| g1gate | s1 | 100M | 0.011 | 0.073 | ~0.21 | 0.845 | 2.04 |
| g1gate | s2 | 100M | 0.0037 | 0.072 | ~0.22 | 0.854 | 2.24 |
| sigmoid | s0 | 102M | 0.830 | 0.377 | — | 1.479 | 1.10 |
| sigmoid | s1 | 100M | 0.756 | 0.311 | — | 1.603 | 1.30 |
| textinit | s0 | 60M | 0.852 | 0.627 | — | 0.377 | 42.5 |
| textinit | s1 | 60M | 0.556 | 0.235 | ~0.53 | 0.634 | 5.50 |
| textinit | s2 | 60M | 0.578 | 0.232 | ~0.59 | 0.484 | 12.2 |

Note on g1gate: both seeds with max-attention data show a single head at ~0.21–0.22 max
attention→pos0 while the arm mean stays ~0.07 and Sink^0.2 stays ≪ 0.05 — reproducible
across seeds, and below every sink threshold used in this paper.

## C. Per-position attention mass (seed 0)

Mean and max-head attention mass by position at seed 0, confirming position 0 is the
maximum-mass token — by mass, not merely argmax count — in every arm. This is the direct
check behind the pos0-anchoring defense of §5.

| arm | mass@pos0 (mean) | max-head@pos0 | argmax-mass position | reading |
|---|---|---|---|---|
| baseline | 0.06 | 0.17 | 0 | pos0 max; diffuse (no sink) |
| g1gate | 0.07 | 0.23 | 0 | pos0 max; diffuse |
| sigmoid | 0.30 | 0.66 | 0 | pos0 max; broad raw-sigmoid mass |
| textinit | 0.63 | 0.99 | 0 | pos0 max; razor spike (pos1 = 0.009) |

Per-position **mass** profiles for seeds 1–2 and RF were not available at the time of
writing (they require a checkpoint re-dump on GPU); only live-probe argmax data exists
there, and it is the source of the textinit position-migration caveat in §5.

## D. Notes on metric hygiene

Two corrections made during the audit of this paper's numbers, recorded for transparency:
(i) an earlier internal consolidation mixed two metrics (mean vs. max attention→pos0) in a
single column, manufacturing an apparent seed anomaly in *g1gate*; re-derived
like-for-like, the anomaly disappears and g1gate is the most reproducible arm.
(ii) An earlier draft claim that textinit's residual-norm peak "stays pinned at pos0" was
checked against the norm dump and found wrong — the ‖h‖ peak sits at pos1 or pos13
depending on seed, which is the positional-decoupling footnote of §3.4, stated there in
its corrected form.
