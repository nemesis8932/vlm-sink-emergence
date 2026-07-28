# Appendix

## A. Supporting figures

<figure id="figA1">
<img src="figures/fig1_layerhead_grid.svg" alt="Per-(layer,head) attention to pos0 over training">
<figcaption><b>Figure A1: Where concentration lives in the network.</b> Attention to pos0
for each (layer, head) pair through training, row-normalized. baseline and RF stay cold
throughout. textinit is hot from initialization. sigmoid lights up a band of mid-network
layers.</figcaption>
</figure>

<figure id="figA2">
<img src="figures/fig3_perhead_scatter.svg" alt="Per-head concentration vs value-norm scatter">
<figcaption><b>Figure A2: No universal head-level coupling.</b> Concentration against
value-norm ratio for each (layer, query-head) pair at the final checkpoint, by arm, at
n = 270 per arm. Under grouped-query attention those pairs hold 90 independent value-norm
observations (&sect;3.3). The sign of the correlation flips by arm, from +0.67 in baseline
to &minus;0.76 in textinit (Table 4). The pooled cloud is weak, at &minus;0.20, only
because arms of opposite sign cancel. Do not read it as an uncorrelated cloud. Most
individual arms show a strong |r|.</figcaption>
</figure>

<figure id="figA3">
<img src="figures/fig5_entropy.svg" alt="Attention entropy over training by arm">
<figcaption><b>Figure A3: Entropy collapse tracks concentration only.</b> Attention entropy
through training. Only sigmoid and textinit collapse. baseline, g1gate, and RF stay flat.
The entropy-collapse correlate from the text-LM literature therefore follows the
concentration axis, not the norm axes.</figcaption>
</figure>

## B. Full per-seed signature table

This table reproduces Table 1 and adds the maximum attention→pos0 where we have it. We
logged that metric for seeds 1 and 2 only, because the seed-0 archive summary does not
include it. We therefore keep it out of the main comparative table.

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

A note on g1gate. Both seeds with max-attention data show one head at about 0.21–0.22
maximum attention→pos0. The arm mean meanwhile stays near 0.07, and Sink^0.2 stays far
below 0.05. That pattern repeats across seeds, and it sits below every sink threshold this paper
uses.

## C. Per-position attention mass (seed 0)

This table gives the mean and max-head attention mass by position at seed 0. It confirms
that position 0 is the maximum-mass token in every arm, by mass rather than by argmax count
alone. It is the direct check behind the position-0 anchoring defense of §5.

| arm | mass@pos0 (mean) | max-head@pos0 | argmax-mass position | reading |
|---|---|---|---|---|
| baseline | 0.06 | 0.17 | 0 | pos0 max; diffuse (no sink) |
| g1gate | 0.07 | 0.23 | 0 | pos0 max; diffuse |
| sigmoid | 0.30 | 0.66 | 0 | pos0 max; broad raw-sigmoid mass |
| textinit | 0.63 | 0.99 | 0 | pos0 max; razor spike (pos1 = 0.009) |

Per-position **mass** profiles for seeds 1 and 2 and for RF were not available when we
wrote this paper. They need a checkpoint re-dump on a GPU. Only live-probe argmax
data exists for those runs, and it is the source of the textinit position-migration caveat
in §5.

## D. Notes on metric hygiene

We record two corrections that we made during the audit of the numbers in this paper.
First, an earlier internal consolidation mixed two metrics in one column, mean and max
attention→pos0, and thereby manufactured an apparent seed anomaly in *g1gate*. Re-derived
like-for-like, the anomaly disappears, and g1gate becomes the most reproducible arm.
Second, an earlier draft claimed that the residual-norm peak of textinit "stays pinned at
pos0." We checked that claim against the norm dump. It is wrong. The ‖h‖ peak sits at
pos1 or at pos13, depending on the seed. That is the positional-decoupling footnote of
§3.4, which states the corrected form.
