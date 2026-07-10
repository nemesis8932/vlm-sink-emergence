#!/usr/bin/env python3
"""Attention-matrix sink-stripe (attn_matrix.npz). The iconic query×key heatmap: a
bright vertical stripe at key=pos0 means every query dumps attention on the first
image token. Top sink head per arm (by attn->pos0), early vs final, plus textinit
at init (the INHERITED sink, present before any VLM training).

All panels row-normalized (each query's attention sums to 1) so the pos0 stripe is
comparable across arms; sigmoid's raw gate is unnormalized and shown normalized here.
"""
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import fig_common as fc

NPZ = 'analysis/attn_matrix.npz'
N_IMG = 49           # 49 image tokens then text (pos0 = first image token)
VMAX = 0.5


def rownorm(m):
    return m / np.clip(m.sum(1, keepdims=True), 1e-9, None)


def pick_strong(M2):
    """M2 (2,128,128) -> the head with the larger row-normalized pos0 share."""
    shares = [rownorm(M2[i])[:, 0].mean() for i in range(M2.shape[0])]
    return M2[int(np.argmax(shares))]


def panel(ax, M, title, color, show_pos0_share=True):
    P = rownorm(M)
    im = ax.imshow(P, aspect='equal', cmap='magma', norm=PowerNorm(0.5, 0, VMAX),
                   interpolation='nearest')
    ax.axvline(0.5, color='cyan', lw=0.7, alpha=0.7)               # pos0 key
    ax.axvline(N_IMG - 0.5, color='white', lw=0.5, alpha=0.4, ls=':')  # img|text boundary
    ax.axhline(N_IMG - 0.5, color='white', lw=0.5, alpha=0.4, ls=':')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8, color=color, weight='bold', pad=2)
    if show_pos0_share:
        share = float(P[:, 0].mean())
        ax.text(0.96, 0.06, f'pos0={share:.2f}', transform=ax.transAxes, fontsize=6.6,
                ha='right', va='bottom', color='cyan')
    return im


def steps_for(pref):
    return sorted(int(re.search(r'step(\d+)', k).group(1))
                  for k in np.load(NPZ).files if k.startswith(pref + '_step'))


def main():
    d = np.load(NPZ)
    # sigmoid column DROPPED (review 2026-07-10): its checkpoint dump selected heads by
    # raw (unnormalized) gate score and missed the arm's true top sink head (L7H3, 0.87
    # normalized) — a panel from the dumped heads visually understates sigmoid's
    # concentration. Caveat lives in the paper prose; sigmoid's concentration is carried
    # by the summary tables + lead-lag figure.
    arms = [('baseline', 'baseline_s0'), ('g1gate', 'g1gate_s0'), ('textinit', 'textinit_s0')]
    fig, axes = plt.subplots(2, 4, figsize=(10.4, 5.9))
    for j, (arm, pref) in enumerate(arms):
        sts = steps_for(pref)
        early, final = sts[0], sts[-1]
        im = panel(axes[0, j], pick_strong(d[f'{pref}_step{early}']), f'{arm}  ·  step {early} (early)',
                   fc.ARM_COLORS[arm])
        panel(axes[1, j], pick_strong(d[f'{pref}_step{final}']), f'{arm}  ·  step {final} (final)',
              fc.ARM_COLORS[arm])
    # 4th column: textinit @ init (inherited) on top, note + colorbar below
    panel(axes[0, 3], pick_strong(d['textinit_s0_step0']), 'textinit · step 0\nINHERITED @init',
          fc.ARM_COLORS['textinit'])
    axes[0, 3].spines[:].set_color(fc.ARM_COLORS['textinit'])
    axes[0, 3].spines[:].set_linewidth(2)
    axes[1, 3].axis('off')
    axes[1, 3].text(0.0, 0.95,
                    'query × key attention\n(top sink head per arm)\n\n'
                    '• bright stripe at key=pos0\n  (cyan line) = sink\n'
                    '• baseline: no stripe\n• textinit: total\n'
                    '• textinit @init already\n  has it (inherited from\n  the text LM)\n\n'
                    'sigmoid omitted: dump\nmissed its true top sink\nhead (L7H3) — see text\n\n'
                    'dotted line = image|text\nboundary (key/query 49)',
                    fontsize=7.2, va='top')
    cax = fig.add_axes([0.80, 0.03, 0.14, 0.022])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label('attn (row-norm, γ=0.5)', fontsize=6.5); cb.ax.tick_params(labelsize=6)

    for i in range(2):
        axes[i, 0].set_ylabel('query position', fontsize=7.5)
    for j in range(4):
        axes[1, j].set_xlabel('key position', fontsize=7.5)
    fig.suptitle('The sink stripe: every query attends to the first image token — '
                 'absent in baseline, total in textinit, and already present at init (inherited)',
                 fontsize=10, weight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=2.2)
    out = 'analysis/fig6_sink_stripe.svg'
    fig.savefig(out, bbox_inches='tight'); fig.savefig(out.replace('.svg', '.png'), dpi=150, bbox_inches='tight')
    print('wrote', out)


if __name__ == '__main__':
    main()
