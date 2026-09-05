#!/usr/bin/env python3
"""Appendix birth-map: per-(layer,head) step of first concentration crossing.

Split out of fig4_birth_leadlag.py when the body figure was reduced to the lead-lag
panel alone for the page limit. Same data, same colour scale, no top panel.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import fig_common as fc
from fig4_birth_leadlag import birth_grid


def main():
    form_arms = ['g1gate', 'sigmoid', 'textinit']
    fig = plt.figure(figsize=(5.5, 3.2))
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    grids = {a: birth_grid(a) for a in form_arms}
    vmax = np.max(np.concatenate([g[~np.isnan(g)] for g in grids.values()]))
    baxes = []
    for k, a in enumerate(form_arms):
        ax = fig.add_subplot(gs[0, k]); baxes.append(ax)
        G = grids[a]
        im = ax.imshow(G, aspect='auto', cmap='viridis_r', norm=Normalize(0, vmax),
                       interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'{a}\n{np.mean(~np.isnan(G))*100:.0f}% cross 0.3',
                     fontsize=8, weight='bold', color=fc.ARM_COLORS[a])
        ax.set_xlabel('9 heads', fontsize=7.5)
        if k == 0:
            ax.set_ylabel('30 layers', fontsize=8)
    cb = fig.colorbar(im, ax=baxes, fraction=0.025, pad=0.02)
    cb.set_label('birth step', fontsize=7); cb.ax.tick_params(labelsize=6.5)
    out = 'analysis/figA4_birthmap.svg'
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace('.svg', '.png'), dpi=150, bbox_inches='tight')
    fig.savefig(out.replace('.svg', '.pdf'), bbox_inches='tight')
    print('wrote', out)


if __name__ == '__main__':
    main()
