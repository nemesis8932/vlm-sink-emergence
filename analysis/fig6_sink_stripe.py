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
    arms = [('baseline', 'baseline_s0'), ('g1gate', 'g1gate_s0'),
            ('sigmoid', 'sigmoid_s0'), ('textinit', 'textinit_s0')]
    fig, axes = plt.subplots(2, 5, figsize=(12.5, 5.4))
    for j, (arm, pref) in enumerate(arms):
        sts = steps_for(pref)
        early, final = sts[0], sts[-1]
        im = panel(axes[0, j], pick_strong(d[f'{pref}_step{early}']), f'{arm}  ·  step {early} (early)',
                   fc.ARM_COLORS[arm])
        panel(axes[1, j], pick_strong(d[f'{pref}_step{final}']), f'{arm}  ·  step {final} (final)',
              fc.ARM_COLORS[arm])
        if arm == 'sigmoid':   # honest flag: dumped heads were raw-selected, miss the true 0.87 head
            axes[1, j].text(0.04, 0.5, 'dumped head\nraw-selected\n(true sink L7H3\n=0.87, not dumped)',
                            transform=axes[1, j].transAxes, fontsize=5.6, color='cyan', va='center')
    # 5th column: textinit @ init (inherited) on top, note + colorbar below
    panel(axes[0, 4], pick_strong(d['textinit_s0_step0']), 'textinit · step 0\nINHERITED @init',
          fc.ARM_COLORS['textinit'])
    axes[0, 4].spines[:].set_color('tab:orange'); axes[0, 4].spines[:].set_linewidth(2)
    axes[1, 4].axis('off')
    axes[1, 4].text(0.0, 0.95,
                    'query × key attention\n(top sink head per arm)\n\n'
                    '• bright stripe at key=pos0\n  (cyan line) = sink\n'
                    '• baseline: no stripe\n• sigmoid/textinit: strong\n'
                    '• textinit @init already\n  has it (inherited from\n  the text LM)\n\n'
                    'dotted line = image|text\nboundary (key/query 49)',
                    fontsize=7.4, va='top')
    cax = fig.add_axes([0.845, 0.08, 0.11, 0.025])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label('attn (row-norm, γ=0.5)', fontsize=6.5); cb.ax.tick_params(labelsize=6)

    for i in range(2):
        axes[i, 0].set_ylabel('query position', fontsize=7.5)
    for j in range(5):
        axes[1, j].set_xlabel('key position', fontsize=7.5)
    fig.suptitle('The sink stripe: every query attends to the first image token — '
                 'absent in baseline, total in textinit, and already present at init (inherited)',
                 fontsize=10.5, weight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = 'analysis/fig6_sink_stripe.svg'
    fig.savefig(out, bbox_inches='tight'); fig.savefig(out.replace('.svg', '.png'), dpi=150, bbox_inches='tight')
    print('wrote', out)


if __name__ == '__main__':
    main()
