#!/usr/bin/env python3
"""Per-head decoupling scatter. Every (layer,head) is a dot: x = attn->pos0,
y = value-norm ratio ‖v‖pos0/‖v‖rest, at each arm's final checkpoint.

Thesis at the head level: if concentration CAUSED value-drain, dots would fall on a
downward line (more attention -> lower v_ratio). Instead each arm is its own cloud and
the pooled relationship is weak -> concentration and value-drain are independently set
per head, not one phenomenon. Pearson r reported per arm + pooled.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import fig_common as fc

ARMS = fc.MAIN_ARMS + ['rf']


def main():
    fig, (ax, axt) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={'width_ratios': [2.1, 1]})
    allx, ally = [], []
    rows = []
    for arm in ARMS:
        d = fc.load_perhead_npz(arm)             # TRUE per-head v_ratio (npz, not per-layer probe)
        x = d['attn_to_pos0'].ravel()
        y = d['v_ratio'].ravel()
        c = fc.ARM_COLORS[arm]
        ax.scatter(x, y, s=14, color=c, alpha=0.55, edgecolors='none', label=fc.ARM_LABELS[arm])
        r, p = pearsonr(x, y)
        rows.append((arm, len(x), float(r), float(p), float(x.max()), float(y.min()), float(y.max())))
        allx.append(x); ally.append(y)
    allx = np.concatenate(allx); ally = np.concatenate(ally)
    rp, pp = pearsonr(allx, ally)

    ax.axhline(1.0, color='black', lw=0.6, alpha=0.4, ls='--')
    ax.set_xlabel('attention concentration  (attn→pos0, per head)', fontsize=9)
    ax.set_ylabel('value-norm ratio  ‖v‖$_{pos0}$ / ‖v‖$_{rest}$  (per head)', fontsize=9)
    ax.set_title(f'Per-head: same concentration → opposite value-norm  (pooled r = {rp:+.2f})',
                 fontsize=10, weight='bold')
    ax.legend(fontsize=7.2, loc='upper left', framealpha=0.9, markerscale=1.6)
    fc.style_ax(ax)
    # call out the two high-concentration populations going OPPOSITE ways
    ax.annotate('sigmoid: high concentration,\nvalue AMPLIFIED (>1)', (0.55, 2.2), fontsize=7.5,
                color='tab:red', ha='left', va='center')
    ax.annotate('textinit: high concentration,\nvalue DRAINED (→0)', (0.55, 0.18), fontsize=7.5,
                color='tab:orange', ha='left', va='center')

    # per-arm r table panel
    axt.axis('off')
    txt = [f'{"arm":10}{"n":>5}{"Pearson r":>11}']
    txt.append('─' * 28)
    for arm, n, r, p, xmx, ymn, ymx in rows:
        sig = '***' if p < 1e-3 else ('*' if p < 0.05 else 'ns')
        txt.append(f'{arm:10}{n:>5}{r:>+9.2f} {sig}')
    txt.append('─' * 28)
    txt.append(f'{"POOLED":10}{len(allx):>5}{rp:>+9.2f} {"***" if pp<1e-3 else "ns"}')
    txt.append('')
    txt.append('the SIGN flips by arm:')
    txt.append('+0.67 baseline … −0.76')
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
    print('\nPer-arm Pearson r (attn->pos0 vs v_ratio, final ckpt):')
    for arm, n, r, p, xmx, ymn, ymx in rows:
        print(f'  {arm:10} n={n:3d} r={r:+.3f} p={p:.1e}  attn_max={xmx:.3f} v_ratio[{ymn:.2f},{ymx:.2f}]')
    print(f'  POOLED     n={len(allx)} r={rp:+.3f} p={pp:.1e}')


if __name__ == '__main__':
    main()
