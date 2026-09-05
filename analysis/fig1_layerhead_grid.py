#!/usr/bin/env python3
"""Layer×head cross-section grid. For each arm, a 30-layer × 9-head heatmap of
attn->pos0 at init / early / final — WHERE in the network the concentration sink
lives and how it forms. From probes.jsonl (per-head attn_to_pos0).

Reading: baseline/RF stay cold everywhere (no sink). sigmoid lights a band of heads.
textinit is hot at init (inherited) and saturates. g1gate develops a few late heads.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.patheffects as pe
import fig_common as fc

ARMS = fc.MAIN_ARMS + ['rf']
VMAX = 0.6   # shared color scale (attn->pos0); textinit/sigmoid saturate, others stay dark


def grid_at(arm, which):
    d = fc.load_perhead(arm, which)
    return d['attn_to_pos0'], d['step']          # (30,9)


def main():
    ncol = len(ARMS)
    fig, axes = plt.subplots(3, ncol, figsize=(5.5, 4.4))
    rows = ['init', 'early', 'final']
    for j, arm in enumerate(ARMS):
        last_step = fc.load_summ(arm)[-1]['step']
        early_step = int(last_step * 0.25)
        steps = {'init': 'first', 'early': early_step, 'final': 'last'}
        for i, rk in enumerate(rows):
            ax = axes[i, j]
            g, st = grid_at(arm, steps[rk])
            im = ax.imshow(g, aspect='auto', cmap='inferno', norm=Normalize(0, VMAX),
                           interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(fc.ARM_LABELS[arm].split(' (')[0], fontsize=7, weight='bold',
                             color=fc.ARM_COLORS[arm])
            if j == 0:
                ax.set_ylabel(f'{rk}\n30 layers', fontsize=7)
            ax.text(0.97, 0.04, f'{st}', transform=ax.transAxes, fontsize=6.5,
                    ha='right', va='bottom', color='white',
                    path_effects=[pe.withStroke(linewidth=1.6, foreground='black')])
    for j in range(ncol):
        axes[2, j].set_xlabel('9 heads', fontsize=7)
    cbar = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.012)
    cbar.set_label('attention to position 0', fontsize=7); cbar.ax.tick_params(labelsize=6)
    cbar.set_ticks([0, .2, .4, .6]); cbar.ax.set_yticklabels(['0', '.2', '.4', '≥.6'])
    out = 'analysis/fig1_layerhead_grid.svg'
    fig.savefig(out, bbox_inches='tight'); fig.savefig(out.replace('.svg', '.png'), dpi=150, bbox_inches='tight')
    fig.savefig(out.replace('.svg', '.pdf'), bbox_inches='tight')
    print('wrote', out)


if __name__ == '__main__':
    main()
