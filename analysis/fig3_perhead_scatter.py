#!/usr/bin/env python3
"""Decoupling scatter. Every (layer, KV-group) is a dot: x = attn->pos0,
y = value-norm ratio ‖v‖pos0/‖v‖rest, at each arm's final checkpoint.

Thesis at the head level: if concentration CAUSED value-drain, dots would fall on a
downward line (more attention -> lower v_ratio). Instead each arm is its own cloud and
the pooled relationship is weak -> concentration and value-drain are independently set
per head, not one phenomenon. Pearson r reported per arm + pooled, descriptively.

Units: one dot = one (layer, KV-group) pair, 30*3 = 90 per arm. Under GQA a value vector is
shared by 3 query heads, so plotting all 270 (layer, query-head) pairs would triplicate every
value observation. Attention is averaged over the 3 query heads of a group. No p-values: the
observations are not independent draws and the arms are not a factorial design.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import fig_common as fc

ARMS = fc.MAIN_ARMS + ['rf']
N_KV = 3          # KV heads per layer
N_Q = 9           # query heads per layer


def collapse_to_kv(a0, vr):
    """(30,9) query-head arrays -> (90,) KV-group observations. v_ratio is constant within a
    group by construction (the value vector is shared), so collapsing removes triplication
    rather than averaging anything away; attention is averaged over the group's query heads."""
    L, H = a0.shape
    g = H // N_KV
    A = a0.reshape(L, N_KV, g).mean(2)
    V = vr.reshape(L, N_KV, g)
    assert np.allclose(V, V[:, :, :1]), 'v_ratio is not constant within a KV group'
    return A.ravel(), V[:, :, 0].ravel()


def main():
    fig, (ax, axt) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={'width_ratios': [2.1, 1]})
    allx, ally = [], []
    rows = []
    for arm in ARMS:
        d = fc.load_perhead_npz(arm)             # TRUE per-head v_ratio (npz, not per-layer probe)
        x, y = collapse_to_kv(d['attn_to_pos0'], d['v_ratio'])
        c = fc.ARM_COLORS[arm]
        ax.scatter(x, y, s=14, color=c, alpha=0.55, edgecolors='none', label=fc.ARM_LABELS[arm])
        r = float(pearsonr(x, y)[0])
        rows.append((arm, len(x), r, float(x.max()), float(y.min()), float(y.max())))
        allx.append(x); ally.append(y)
    allx = np.concatenate(allx); ally = np.concatenate(ally)
    rp = float(pearsonr(allx, ally)[0])

    ax.axhline(1.0, color='black', lw=0.6, alpha=0.4, ls='--')
    ax.set_xlabel('attention concentration  (attn→pos0, mean over the group\'s query heads)', fontsize=9)
    ax.set_ylabel('value-norm ratio  ‖v‖$_{pos0}$ / ‖v‖$_{rest}$  (per KV group)', fontsize=9)
    ax.set_title(f'Per KV group: same concentration → opposite value-norm  (pooled r = {rp:+.2f})',
                 fontsize=10, weight='bold')
    ax.legend(fontsize=7.2, loc='upper left', framealpha=0.9, markerscale=1.6)
    fc.style_ax(ax)
    # call out the two high-concentration populations going OPPOSITE ways
    box = dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.8)
    ax.annotate('sigmoid: high concentration,\nvalue AMPLIFIED (>1)', (0.62, 2.45), fontsize=7.5,
                color=fc.ARM_COLORS['sigmoid'], ha='left', va='center', weight='bold', bbox=box)
    ax.annotate('textinit: high concentration,\nvalue DRAINED (→0)', (0.52, 0.62), fontsize=7.5,
                color=fc.ARM_COLORS['textinit'], ha='left', va='center', weight='bold', bbox=box)

    # per-arm r table panel
    axt.axis('off')
    txt = [f'{"arm":10}{"n":>5}{"Pearson r":>11}']
    txt.append('─' * 28)
    for arm, n, r, xmx, ymn, ymx in rows:
        txt.append(f'{arm:10}{n:>5}{r:>+9.2f}')
    txt.append('─' * 28)
    txt.append(f'{"POOLED":10}{len(allx):>5}{rp:>+9.2f}')
    txt.append('')
    txt.append('one dot = one (layer, KV')
    txt.append('group); 90 per arm. r is')
    txt.append('descriptive, no p-values.')
    txt.append('')
    txt.append('the SIGN flips by arm:')
    txt.append(f'{rows[0][2]:+.2f} baseline … {rows[3][2]:+.2f}')
    txt.append('textinit. No universal')
    txt.append('head-level law links')
    txt.append('concentration to value-')
    txt.append('norm — the lever sets it.')
    axt.text(0.0, 0.98, '\n'.join(txt), family='monospace', fontsize=8.5, va='top')

    fig.suptitle('No universal coupling: the concentration–value-norm sign is arm-dependent',
                 fontsize=11, weight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = 'analysis/fig3_perhead_scatter.svg'
    fig.savefig(out, bbox_inches='tight'); fig.savefig(out.replace('.svg', '.png'), dpi=160, bbox_inches='tight')
    print('wrote', out)
    print('\nPer-arm Pearson r (attn->pos0 vs v_ratio, final ckpt, 90 KV groups):')
    for arm, n, r, xmx, ymn, ymx in rows:
        print(f'  {arm:10} n={n:3d} r={r:+.3f}  attn_max={xmx:.3f} v_ratio[{ymn:.2f},{ymx:.2f}]')
    print(f'  POOLED     n={len(allx)} r={rp:+.3f}')


if __name__ == '__main__':
    main()
