#!/usr/bin/env python3
"""Sink birth-map + lead-lag ordering.

TOP  — lead-lag timeline: for each arm, the token-budget at which each signature first
crosses threshold (massive-act h_ratio>2, value-drain v_ratio<0.8, concentration
max attn->pos0 >0.3). The thesis as ordering: in the softmax-scratch arms the NORM
signatures cross early and concentration never arrives; sigmoid is the mirror image
(concentration crosses, norms never); textinit has all three at init (inherited).

BOTTOM — birth-map: per-(layer,head) step-of-first-crossing of attn->pos0>0.3, for the
arms that DO form concentration. baseline/RF never cross (no panel).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import fig_common as fc

ARMS = fc.MAIN_ARMS + ['rf']
EPS_CONC = 0.3
FLOOR_M = 0.04   # 'at init' (0 tok) plotted here on the log axis


def first_cross_tokens(S, key, thr, gt=True):
    for s in S:
        v = s[key]
        if (v > thr) if gt else (v < thr):
            return max(s['tokens_seen'] / 1e6, FLOOR_M)
    return None


def birth_grid(arm, eps=EPS_CONC):
    """(30,9) first step attn_norm>eps per head; nan = never."""
    rd = fc.RUN_DIR[arm]
    import json
    P = [json.loads(l) for l in open(f'{rd}/probes.jsonl')]
    P.sort(key=lambda p: p['step'])
    G = np.full((30, 9), np.nan)
    for p in P:
        st = p['step']
        for i in range(30):
            a = p['layers'][i]['attn_to_pos0_norm']
            for h in range(9):
                if np.isnan(G[i, h]) and a[h] > eps:
                    G[i, h] = st
    return G


def main():
    fig = plt.figure(figsize=(11, 7.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.05, 1.0], hspace=0.42, wspace=0.28)

    # ---- TOP: lead-lag timeline ----
    axT = fig.add_subplot(gs[0, :])
    # signature colors deliberately OFF the arm palette (magenta/teal/black) so a
    # marker's hue can't be misread as an arm identity
    sig_styles = [('h_ratio_pos0', 2.0, True, '^', 'massive-act  ‖h‖>2', '#B5487A'),
                  ('v_ratio_pos0', 0.8, False, 's', 'value-drain  ‖v‖<0.8', '#0F7F8B'),
                  ('max_attn_pos0', EPS_CONC, True, '*', 'concentration  attn→pos0>0.3', '#111111')]
    yticks, ylabels = [], []
    for yi, arm in enumerate(ARMS):
        S = fc.load_summ(arm)
        y = len(ARMS) - 1 - yi
        yticks.append(y); ylabels.append(fc.ARM_LABELS[arm].split(' (')[0])
        crosses = []
        never_cols = []   # signatures that never cross -> placed in the "never" band, dodged
        for key, thr, gt, mk, lab, col in sig_styles:
            t = first_cross_tokens(S, key, thr, gt)
            if t is not None:
                crosses.append(t)
                ms = 240 if mk == '*' else 110
                axT.scatter([t], [y], marker=mk, s=ms, color=col, zorder=5,
                            edgecolors='white', linewidths=0.6)
            else:
                never_cols.append((mk, col))
        if crosses:
            axT.plot([min(crosses), max(crosses)], [y, y], color='gray', lw=0.8, alpha=0.4, zorder=1)
        for di, (mk, col) in enumerate(never_cols):   # dodge multiple "never" markers horizontally
            xx = 1450 * (1.45 ** di)
            ms = 240 if mk == '*' else 110
            axT.scatter([xx], [y], marker=mk, s=ms, color=col, zorder=5,
                        edgecolors='white', linewidths=0.6, alpha=0.85)
    axT.set_yticks(yticks); axT.set_yticklabels(ylabels, fontsize=8.5)
    for tick, arm in zip(axT.get_yticklabels(), ARMS):
        tick.set_color(fc.ARM_COLORS[arm]); tick.set_fontweight('bold')
    axT.set_xscale('log'); axT.set_xlim(FLOOR_M * 0.8, 5000); axT.set_ylim(-0.8, len(ARMS) - 0.2)
    axT.set_xlabel('tokens at first threshold-crossing (millions, log)', fontsize=9)
    axT.axvspan(1200, 5000, color='gray', alpha=0.08)
    axT.text(2600, len(ARMS) - 0.35, 'NEVER\ncrosses', fontsize=7.8, color='gray', ha='center',
             va='top', weight='bold')
    axT.set_title('Lead–lag: which sink signature crosses first', fontsize=10, weight='bold')
    axT.text(FLOOR_M, -0.62, '↑ "@init" (0 tok, inherited from text LM)', fontsize=6.8,
             color='dimgray', ha='left')
    fc.style_ax(axT)
    handles = [Line2D([0], [0], marker=mk, color=col, lw=0, markersize=10 if mk == '*' else 8,
                      markeredgecolor='white', label=lab) for _, _, _, mk, lab, col in sig_styles]
    axT.legend(handles=handles, fontsize=7.6, loc='upper left', frameon=True, framealpha=0.9)

    # ---- BOTTOM: birth-maps for arms that form concentration ----
    form_arms = ['g1gate', 'sigmoid', 'textinit']
    # shared color scale over actual crossing steps
    allvals = []
    grids = {}
    for a in form_arms:
        G = birth_grid(a); grids[a] = G
        allvals.append(G[~np.isnan(G)])
    vmax = np.nanpercentile(np.concatenate(allvals), 98) if allvals else 1
    baxes = []
    for k, a in enumerate(form_arms):
        ax = fig.add_subplot(gs[1, k]); baxes.append(ax)
        G = grids[a]
        im = ax.imshow(G, aspect='auto', cmap='viridis_r', norm=Normalize(0, vmax),
                       interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        frac = np.mean(~np.isnan(G))
        ax.set_title(f'{a}\n{frac*100:.0f}% of heads cross ε=0.3', fontsize=8.5,
                     weight='bold', color=fc.ARM_COLORS[a])
        ax.set_xlabel('9 heads', fontsize=7.5)
        if k == 0:
            ax.set_ylabel('30 layers', fontsize=8)
    # ONE shared colorbar for the three birth-maps (same Normalize) instead of three copies
    cb = fig.colorbar(im, ax=baxes, fraction=0.025, pad=0.02)
    cb.set_label('birth step', fontsize=7); cb.ax.tick_params(labelsize=6.5)
    axn = fig.add_subplot(gs[1, 3]); axn.axis('off')
    axn.text(0.02, 0.92, 'birth-map\n(step a head first\ncrosses attn→pos0 > 0.3)',
             fontsize=8.5, weight='bold', va='top')
    axn.text(0.02, 0.55,
             'baseline:  0 % of heads\n             ever cross ε=0.3\n\n'
             'RF (fresh): 0 % of heads\n             ever cross ε=0.3\n\n'
             '→ the concentration sink\n   never forms from random\n   init (repeated OR fresh).',
             fontsize=8, va='top', color='dimgray')

    fig.suptitle('Sink birth & ordering: norm signatures lead; concentration is late, mirror-imaged, or never',
                 fontsize=11, weight='bold', y=0.98)
    out = 'analysis/fig4_birth_leadlag.svg'
    fig.savefig(out, bbox_inches='tight'); fig.savefig(out.replace('.svg', '.png'), dpi=150, bbox_inches='tight')
    print('wrote', out)


if __name__ == '__main__':
    main()
