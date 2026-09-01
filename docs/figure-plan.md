# Figure plan for the VLM4RWD submission

Where to start if you want the paper to look like the well-drawn ones, and which figures
earn their page space. Written 2026-09-02 against the 8-page body.

## What is wrong with the current figures, concretely

Every figure was drawn at 11-13 inches wide with 7-10pt fonts, then scaled to a 5.5-inch
text block. That multiplies every font by about 0.45, so a 7.5pt label prints at 3.4pt.
Fixed for Fig 1 and Fig 2 in this pass: they are now drawn at 5.5in with 6-7.5pt fonts, so
what you see in matplotlib is what prints. The appendix figures (A1-A5) still need the same
treatment. It is a 10-line change per script: `figsize`, the font sizes, drop the
`suptitle` (the LaTeX caption carries the title now), and re-run.

Rule for every new figure: **draw at the width it will print at.** 5.5in for full width,
2.65in for a half-width pair. Nothing under 6pt. No suptitle.

## The one diagram to make yourself: the setup figure

This is the highest-value addition and the one that shortens the paper. Section 3.1 spends
a paragraph and a five-column table saying what a single drawing says faster. Target: it
replaces the arms table and about 120 words of prose, and becomes the new Figure 1.

Sketch, left to right:

```
 image (SigLIP-B/16, pretrained) ──► 49 image tokens ─┐
                                                       ├──► [ decoder: SmolLM2-135M arch, 30 layers ]
 text ─────────────────────────────► 79 text tokens ──┘        │
                                                               ▼
   ┌── pos 0 = first image token, no BOS ◄───── probe every 100 steps
   │   three signatures read here:
   │     concentration  Sink^ε  (270 query heads)
   │     v-ratio               (90 KV groups)
   │     h-ratio               (30 layers)
   │
   └── four levers, one knob each (everything else byte-identical)
         baseline   softmax, random init
         g1gate     softmax + zero-init σ-gate on the output
         sigmoid    sigmoid, no softmax
         textinit   softmax, decoder loaded from SmolLM2
       + RF: baseline recipe, fresh 1B-token stream (2.39 visual epochs)
```

What makes it work, in order of importance:

1. **Position 0 is the visual anchor of the drawing.** Draw the token row, color the first
   box, and hang the three signature readouts off it. A reader who understands nothing else
   should see "they measure three things at the first image token."
2. **One glyph per lever on the decoder block**, not a table. A small icon where each lever
   acts: the gate sits on the attention output, sigmoid replaces the softmax box, textinit
   is an arrow loading weights into the decoder, baseline is the plain block. RF is the same
   block with a different data pipe feeding it.
3. **The three granularities as small numbers (270 / 90 / 30)** next to each readout. That
   single detail pre-empts the pseudoreplication question in section 4.3.
4. Keep the arm colors identical to Fig 1 and Fig 2 (`analysis/fig_common.py`,
   `ARM_COLORS`). Same blue, green, red, orange, violet everywhere. The reader learns the
   palette once.

Tools: draw it in whatever you are fastest in (Keynote, Figma, Excalidraw, draw.io) and
export **PDF with fonts embedded**, not PNG. Check with `pdffonts` that no Type 3 font
appears. Use DejaVu Sans or Helvetica so it matches the plots. 5.5in wide, about 2.2in tall.

If you would rather I draw it: matplotlib can do this with `patches` and `annotate`, and
it stays vector and in the same palette. About 150 lines. Say the word.

## Second candidate: a "three axes" glyph for the abstract or the intro

A tiny three-axis schematic (concentration / value-norm / residual-norm) with the four arm
markers placed at their Table 1 corners. 2.65in, half-width, floated beside the
contributions paragraph. It is what Fig 1 shows, without the trajectories, and it lets a
skimming reader get the "four corners" claim from page 1. Optional. Only if the setup figure
lands and there is still space.

## Where the existing figures should live

| figure | now | keep? | why |
|---|---|---|---|
| Fig 1 phase portrait | body, full width | **yes, hero** | it is the thesis as motion |
| Fig 2 sink stripe | body, full width | yes | the textinit-at-step-0 panel is the single most persuasive image in the paper |
| A1 layer-head grid | appendix | yes | supports 4.1, too dense for the body |
| A2 per-head scatter | appendix | yes | the descriptive-correlation evidence behind Table 3 |
| A3 entropy | appendix | yes | one-line supporting correlate |
| A4 lead-lag | appendix | yes | the timing claim in 4.1 rests on it |
| A5 birth-map | appendix | maybe drop | A1 already shows where concentration lives; A5 adds *when*, which A4 gives per arm |

## Fixes still owed on the existing figures

- **Fig 2, fifth column, lower panel**: the free-text legend is now at 4.6pt to fit. Move
  that explanation into the caption and delete the text panel, then use the space for a
  proper colorbar with a label. That is the last illegible thing in the body.
- **Appendix figures A1-A5**: same print-size treatment as Fig 1 and Fig 2.
- **Fig 1 right panel**: the y-label reads "residual-norm ratio" now (was
  "massive-activation ratio"), matching the proxy language in the text. Check A-figures for
  the same wording drift.

## Style references worth stealing from

- Gu et al. (ICLR 2025, arXiv:2410.10781): Figure 1 is a single attention heatmap with the
  sink column annotated. One image, one idea.
- Sun et al. (COLM 2024, arXiv:2402.17762): Figure 1 shows a handful of activation
  magnitudes as bars with the massive ones towering. Pure contrast, no legend needed.
- Chen and Yao (arXiv:2603.17771): a two-panel forward/backward diagram of where the
  gradient concentrates. Closest in spirit to the setup figure above.

The common thread: the first figure in each paper shows the *object* (a sink, a massive
activation, a gradient) before it shows any result. Ours currently opens with a result. The
setup figure fixes that.
