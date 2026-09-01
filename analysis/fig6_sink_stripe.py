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


def true_share(run_dir, step, layer, head):
    """norm_to_pos0 from the reprobe dump: masked to valid queries, as the paper reports."""
    import os
    p = f'{run_dir}/reprobe/reprobe_step{step}.npz'
    if not os.path.exists(p):
        return None
    return float(np.load(p)['norm_to_pos0'][layer, head])


def panel(ax, M, title, color, show_pos0_share=True, share_override=None):
    P = rownorm(M)
    im = ax.imshow(P, aspect='equal', cmap='magma', norm=PowerNorm(0.5, 0, VMAX),
                   interpolation='nearest')
    ax.axvline(0.5, color='cyan', lw=0.7, alpha=0.7)               # pos0 key
    ax.axvline(N_IMG - 0.5, color='white', lw=0.5, alpha=0.4, ls=':')  # img|text boundary
    ax.axhline(N_IMG - 0.5, color='white', lw=0.5, alpha=0.4, ls=':')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=5.2, color=color, weight='bold', pad=2, linespacing=1.2)
    if show_pos0_share:
        share = float(P[:, 0].mean()) if share_override is None else share_override
        ax.text(0.96, 0.06, f'pos0={share:.2f}', transform=ax.transAxes, fontsize=5,
                ha='right', va='bottom', color='cyan')
    return im


def steps_for(pref):
    return sorted(int(re.search(r'step(\d+)', k).group(1))
                  for k in np.load(NPZ).files if k.startswith(pref + '_step'))


def main():
    d = np.load(NPZ)
    # sigmoid column RESTORED (2026-08-10): re-dumped its true top sink head L7H3
    # (0-indexed; norm_to_pos0 = 0.873 at the final ckpt, the arm max) with
    # analysis/dump_sigmoid_l7h3.py. The original dump selected heads by raw
    # (unnormalized) gate score and picked L8H3/L1H5, which understated the arm.
    arms = [('baseline', 'baseline_s0'), ('g1gate', 'g1gate_s0'),
            ('sigmoid', 'sigmoid_s0'), ('textinit', 'textinit_s0')]
    # sized to the NeurIPS 5.5in text block so every font prints at its nominal size
    fig, axes = plt.subplots(2, 5, figsize=(5.5, 2.75))
    fig.subplots_adjust(left=0.05, right=0.995, top=0.90, bottom=0.09, wspace=0.12, hspace=0.42)
    SIG_LH = (7, 3)   # 0-indexed: the sigmoid arm's true top sink head
    TOK = {('baseline', 250): '2.4M', ('baseline', 18287): '174M',
           ('g1gate', 250): '2.4M', ('g1gate', 10786): '103M',
           ('sigmoid', 250): '2.4M', ('sigmoid', 10664): '102M',
           ('textinit', 250): '2.4M', ('textinit', 6244): '60M', ('textinit', 0): '0'}
    for j, (arm, pref) in enumerate(arms):
        if arm == 'sigmoid':
            early, final = 250, 10664
            Me = d[f'{pref}_l7h3_step{early}']; Mf = d[f'{pref}_l7h3_step{final}']
            se = true_share('runs/sigmoid', early, *SIG_LH)
            sf = true_share('runs/sigmoid', final, *SIG_LH)
            tag = f' L{SIG_LH[0]}H{SIG_LH[1]}'
        else:
            sts = [s for s in steps_for(pref) if s > 0] if arm == 'textinit' else steps_for(pref)
            early, final = sts[0], sts[-1]
            Me = pick_strong(d[f'{pref}_step{early}']); Mf = pick_strong(d[f'{pref}_step{final}'])
            se = sf = None
            tag = ''
        # short two-line titles: the caption carries "seed 0" and the step numbers live in
        # Appendix A, so each panel only needs arm, phase and token count
        im = panel(axes[0, j], Me, f'{arm}{tag}\nearly · {TOK.get((arm, early), "?")} tok',
                   fc.ARM_COLORS[arm], share_override=se)
        panel(axes[1, j], Mf, f'{arm}{tag}\nfinal · {TOK.get((arm, final), "?")} tok',
              fc.ARM_COLORS[arm], share_override=sf)
    # 5th column: textinit @ init (inherited) on top, note + colorbar below
    panel(axes[0, 4], pick_strong(d['textinit_s0_step0']), 'textinit at init\n0 tok · inherited',
          fc.ARM_COLORS['textinit'])
    axes[0, 4].spines[:].set_color(fc.ARM_COLORS['textinit'])
    axes[0, 4].spines[:].set_linewidth(2)
    # the explanatory text lives in the LaTeX caption; the free cell holds the colorbar
    axes[1, 4].axis('off')
    pos = axes[1, 4].get_position()
    cax = fig.add_axes([pos.x0 + 0.15 * pos.width, pos.y0 + 0.45 * pos.height,
                        0.7 * pos.width, 0.06 * pos.height])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label('attention (row-normalized, γ = 0.5)', fontsize=5.5, labelpad=2)
    cb.ax.tick_params(labelsize=5, length=2, pad=1)

    for i in range(2):
        axes[i, 0].set_ylabel('query position', fontsize=5.5)
    for j in range(4):
        axes[1, j].set_xlabel('key position', fontsize=5.5)
    axes[0, 4].set_xlabel('key position', fontsize=5.5)
    # no suptitle: the LaTeX caption carries the title
    out = 'analysis/fig6_sink_stripe.svg'
    fig.savefig(out, bbox_inches='tight'); fig.savefig(out.replace('.svg', '.png'), dpi=150, bbox_inches='tight')
    fig.savefig(out.replace('.svg', '.pdf'), bbox_inches='tight')
    print('wrote', out)


if __name__ == '__main__':
    main()
