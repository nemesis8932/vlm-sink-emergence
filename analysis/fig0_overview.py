#!/usr/bin/env python3
"""Figure 1, the overview: what we train, where we measure, and the four corners.

Left  : the pipeline. Image -> 49 image tokens, text -> 79 text tokens, into a randomly
        initialized SmolLM2-135M-architecture decoder. Position 0 is the first image token
        (no BOS). A probe reads three signatures there every 100 steps. Four levers, one
        chip each; RF is the same recipe on a fresh 1B-token stream.
Right : the four corners of Table 1 as a fingerprint strip: one row per arm, one column per
        signature, the per-arm seed range as a bar and its midpoint as a marker.

Drawn at the NeurIPS text width (5.5in). The left axes maps one data unit to one inch, so
the coordinates below are inches on the page and font sizes are true points.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import fig_common as fc

C = fc.ARM_COLORS
INK, MUTE, PALE = '#222222', '#666666', '#f2f2f2'
W, H = 5.5, 3.05            # figure inches
LW = 3.45                    # left panel width in inches


def box(ax, x, y, w, h, text, fc_='white', ec=INK, lw=0.8, fs=6.5, weight='normal',
        color=INK, ls='-'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.05',
                                fc=fc_, ec=ec, lw=lw, ls=ls, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=color,
            weight=weight, zorder=4, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, color=INK, lw=0.8, ms=6):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=ms,
                                 color=color, lw=lw, zorder=5, shrinkA=0, shrinkB=0))


def pipeline(ax):
    ax.set_xlim(0, LW); ax.set_ylim(0, H); ax.axis('off'); ax.set_aspect('equal')
    ax.text(0.04, H - 0.06, 'A  what we train, where we measure', fontsize=6.5,
            weight='bold', va='top')

    # ---- inputs (two rows) -> encoders
    box(ax, 0.05, 2.15, 0.52, 0.34, 'image', fc_=PALE, ec='#bbbbbb')
    box(ax, 0.05, 1.62, 0.52, 0.34, 'text', fc_=PALE, ec='#bbbbbb')
    box(ax, 0.70, 2.15, 0.82, 0.34, 'SigLIP-B/16\n(pretrained)', fs=5.6)
    box(ax, 0.70, 1.62, 0.82, 0.34, 'tokenizer', fs=5.6)
    arrow(ax, 0.57, 2.32, 0.70, 2.32); arrow(ax, 0.57, 1.79, 0.70, 1.79)

    # ---- token row
    s, g = 0.050, 0.014
    x0, y0 = 1.66, 1.93
    n_img, n_txt = 11, 7
    for i in range(n_img):
        ax.add_patch(Rectangle((x0 + i * (s + g), y0), s, s,
                               fc=C['textinit'] if i == 0 else '#8fb3d9', ec='white', lw=0.4, zorder=3))
    xt = x0 + n_img * (s + g) + 0.04
    for i in range(n_txt):
        ax.add_patch(Rectangle((xt + i * (s + g), y0), s, s, fc='#c9c9c9', ec='white', lw=0.4, zorder=3))
    xend = xt + n_txt * (s + g)
    ax.text(x0 + n_img * (s + g) / 2, y0 + s + 0.04, '49 image tokens', fontsize=5.4,
            ha='center', va='bottom', color=MUTE)
    ax.text(xt + n_txt * (s + g) / 2, y0 + s + 0.04, '79 text', fontsize=5.4,
            ha='center', va='bottom', color=MUTE)
    arrow(ax, 1.52, 2.32, x0 - 0.02, y0 + s + 0.02, lw=0.6, ms=4)
    arrow(ax, 1.52, 1.79, x0 - 0.02, y0 + s / 2, lw=0.6, ms=4)
    # position-0 callout, under the row
    ax.annotate('pos 0 = first image token\n(no BOS)', xy=(x0 + s / 2, y0), xytext=(x0, 1.70),
                fontsize=5.6, color=C['textinit'], weight='bold', ha='left', va='top',
                zorder=6, linespacing=1.15,
                arrowprops=dict(arrowstyle='-', color=C['textinit'], lw=0.6, shrinkA=0, shrinkB=1))

    # ---- decoder
    dx, dy, dw, dh = 2.62, 1.05, 0.80, 1.44
    ax.add_patch(FancyBboxPatch((dx, dy), dw, dh, boxstyle='round,pad=0,rounding_size=0.05',
                                fc='white', ec=INK, lw=0.9, zorder=3))
    ax.text(dx + dw / 2, dy + dh - 0.07, 'decoder', ha='center', va='top', fontsize=6.8,
            weight='bold', zorder=4)
    ax.text(dx + dw / 2, dy + dh - 0.22, 'SmolLM2-135M arch\n30 layers, GQA 9q/3kv\nrandom init',
            ha='center', va='top', fontsize=5.0, color=MUTE, zorder=4, linespacing=1.3)
    arrow(ax, xend + 0.02, y0 + s / 2, dx - 0.01, y0 + s / 2)
    # lever chips
    chips = [('baseline', 'softmax'), ('g1gate', '+ σ-gate, zero-init'),
             ('sigmoid', 'sigmoid, no softmax'), ('textinit', 'SmolLM2 weights')]
    ax.text(dx + 0.06, dy + 0.72, 'one lever per arm', fontsize=5.0, color=MUTE, va='bottom')
    for k, (arm, lab) in enumerate(chips):
        yy = dy + 0.60 - k * 0.15
        ax.add_patch(Rectangle((dx + 0.06, yy - 0.045), 0.045, 0.09, fc=C[arm], ec='none', zorder=4))
        ax.text(dx + 0.135, yy, lab, fontsize=5.0, va='center', color=INK, zorder=4)

    # ---- RF strip under the decoder
    box(ax, dx, 0.42, dw, 0.50, 'RF\nbaseline recipe,\nfresh 1B-token stream\n2.39 visual epochs',
        fc_='#f1eef7', ec=C['rf'], fs=4.9, color=C['rf'])

    # ---- probe readouts
    px, py, pw, ph = 0.05, 0.10, 2.42, 1.30
    ax.add_patch(FancyBboxPatch((px, py), pw, ph, boxstyle='round,pad=0,rounding_size=0.05',
                                fc='#fbf6ee', ec=C['textinit'], lw=0.8, ls=(0, (3, 2)), zorder=3))
    ax.text(px + 0.08, py + ph - 0.08, 'probe at pos 0, every 100 optimizer steps',
            fontsize=6, weight='bold', va='top', zorder=4)
    rows = [('concentration', r'Sink$^{\epsilon}$ = share of heads with attn$\to$pos0 $> \epsilon$', '270 heads'),
            ('v-ratio', r'$\|v_0\|\,/\,\overline{\|v_{i>0}\|}$,  drain $<1<$ amplified', '90 KV groups'),
            ('h-ratio', r'$\|h_0\|\,/\,\overline{\|h_{i>0}\|}$,  massive-activation proxy', '30 layers')]
    for k, (name, form, n) in enumerate(rows):
        yy = py + ph - 0.40 - k * 0.30
        ax.text(px + 0.08, yy + 0.06, name, fontsize=5.6, weight='bold', va='center', zorder=4)
        ax.text(px + 0.08, yy - 0.09, form, fontsize=5.0, va='center', zorder=4)
        ax.text(px + pw - 0.08, yy + 0.06, n, fontsize=5.0, va='center', ha='right', color=MUTE, zorder=4)
    arrow(ax, x0 + s / 2, 1.52, px + 1.4, py + ph, color=C['textinit'], lw=0.6, ms=5)


def corners(ax):
    arms = ['baseline', 'g1gate', 'sigmoid', 'textinit', 'rf']
    conc = {'baseline': (0, 0), 'g1gate': (0.004, 0.011), 'sigmoid': (0.76, 0.83),
            'textinit': (0.56, 0.85), 'rf': (0, 0)}
    vr = {'baseline': (0.69, 0.72), 'g1gate': (0.81, 0.85), 'sigmoid': (1.48, 1.60),
          'textinit': (0.38, 0.63), 'rf': (0.69, 0.69)}
    hr = {'baseline': (1.7, 2.2), 'g1gate': (1.7, 2.2), 'sigmoid': (1.1, 1.3),
          'textinit': (5.5, 42.5), 'rf': (3.22, 3.22)}
    cols = [('conc.\nSink$^{0.2}$', conc, (0, 1), False, [0, 0.5, 1]),
            ('v-ratio', vr, (0.3, 1.7), False, [0.5, 1, 1.5]),
            ('h-ratio\n(log)', hr, (1, 50), True, [1, 3, 10, 40])]
    n = len(arms)
    ax.set_xlim(-0.62, 3.0); ax.set_ylim(-0.95, n + 0.9); ax.axis('off')
    ax.text(-0.62, n + 0.85, 'B  the four corners (Table 1)', fontsize=6.5, weight='bold',
            va='top', ha='left')
    for j, (title, d, (lo, hi), log, ticks) in enumerate(cols):
        xa, xb = j + 0.10, j + 0.92
        f = (lambda v, lo=lo, hi=hi: (np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo))) if log \
            else (lambda v, lo=lo, hi=hi: (v - lo) / (hi - lo))
        ax.text((xa + xb) / 2, n - 0.35, title, ha='center', va='bottom', fontsize=5.4,
                weight='bold', linespacing=1.15)
        for i, arm in enumerate(arms):
            y = n - 1 - i
            ax.plot([xa, xb], [y, y], color='#dddddd', lw=0.6, zorder=1)
            a, b = d[arm]
            pa, pb = xa + f(max(a, lo)) * (xb - xa), xa + f(max(b, lo)) * (xb - xa)
            if pb - pa > 0.01:
                ax.plot([pa, pb], [y, y], color=C[arm], lw=3, alpha=0.35,
                        solid_capstyle='round', zorder=2)
            ax.plot((pa + pb) / 2, y, marker='D' if arm != 'rf' else 'o', ms=3.4,
                    color=C[arm], mec='white', mew=0.5, zorder=3)
        if j == 1:
            xr = xa + f(1.0) * (xb - xa)
            ax.plot([xr, xr], [-0.4, n - 0.55], color=MUTE, lw=0.5, ls=(0, (2, 2)), zorder=1)
        for v in ticks:
            ax.text(xa + f(v) * (xb - xa), -0.5, f'{v:g}' + ('×' if log else ''),
                    ha='center', va='top', fontsize=4.6, color=MUTE)
    for i, arm in enumerate(arms):
        ax.text(-0.02, n - 1 - i, 'RF' if arm == 'rf' else arm, ha='right', va='center',
                fontsize=5.4, color=C[arm], weight='bold')
    ax.text(1.2, -0.85, 'bar = seed range (n = 2–3), marker = midpoint\nRF: one run, read at 1B tokens',
            ha='center', va='top', fontsize=4.6, color=MUTE, linespacing=1.2)


def main():
    fig = plt.figure(figsize=(W, H))
    axL = fig.add_axes([0.0, 0.0, LW / W, 1.0])
    axR = fig.add_axes([LW / W + 0.02, 0.0, 1 - LW / W - 0.02, 1.0])
    pipeline(axL); corners(axR)
    out = 'analysis/fig0_overview.svg'
    fig.savefig(out, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(out.replace('.svg', '.png'), dpi=220, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(out.replace('.svg', '.pdf'), bbox_inches='tight', pad_inches=0.02)
    print('wrote', out)


if __name__ == '__main__':
    main()
