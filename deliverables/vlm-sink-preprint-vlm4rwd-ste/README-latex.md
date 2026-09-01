# NeurIPS 2026 LaTeX build

The markdown in `sections/` stays the single source. Two backends read it:

| backend | driver | output | purpose |
|---|---|---|---|
| HTML/PDF reading copy | `build.py` | `paper-v2-ste.pdf` | fast to read, not a submission format |
| NeurIPS LaTeX | `build_tex.py` → `make_pdf.py` | `paper-neurips.pdf` | the actual submission |

Edit prose only in `sections/*.md`. Never hand-edit the generated `.tex`.

## Status

Built and verified against the real `neurips_2026.sty` (revision 2026-01-29):
**body = 8 pages**, 21 pages total, 0 overfull boxes, no Type 3 fonts. The submission build is
anonymous with line numbers; the preprint build is named with the "Preprint." footer.

## Setup, if starting from a clean machine

**1. Install pdflatex.** The call for papers says to generate the PDF with `pdflatex`.

Note: do not paste these with trailing `#` comments. zsh does not treat `#` as a comment
in an interactive shell by default, so the comment text is passed to brew as arguments.

```
brew install --cask basictex
```

Then open a new terminal window, and:

```
sudo tlmgr update --self
sudo tlmgr install type1cm cm-super environ trimspaces helvetic courier
```

`booktabs`, `microtype`, `natbib`, `tabularx`, `geometry`, `lineno` and `times` already ship
with BasicTeX. `environ` is required by the style file itself; `helvetic` and `courier` supply
the Helvetica and Courier metrics the style file loads. `nicefrac` is
not a package of its own (it lives in `units`) and the paper never uses it, so it was dropped
from the preamble.

**2. The style file.** `neurips_2026.sty` is committed here, unmodified
(`2026-01-29 NeurIPS 2026 submission/camera-ready style file`). Do not edit it — "Tweaking
the style files may be grounds for desk rejection."

Then:

```
python3 make_pdf.py              # anonymous submission build, with line numbers
python3 make_pdf.py --preprint   # named arXiv build
```

`make_pdf.py` reports total pages, body pages, and whether any Type 3 font slipped in. It adds
`/Library/TeX/texbin` to PATH itself, so a fresh BasicTeX install needs no terminal restart.

Without the style file you can still check that the generated LaTeX is valid:

```
python3 make_pdf.py --smoketest
```

That compiles against plain `article` with the style file's macros stubbed out, so a failure
is a converter bug rather than a missing `.sty`. Current status: 0 errors, 0 overfull boxes,
0 undefined citations. The page count from that build is meaningless, since the geometry is
not NeurIPS geometry.

## What the build enforces from the formatting instructions

- **Options.** Submission passes no option, so the style file anonymises and adds line
  numbers. `--preprint` passes `[preprint]`. `[final]` is never emitted; it is only for
  accepted papers.
- **Abstract** is a single paragraph.
- **Headings** are lower case except the first word and proper nouns (`lower_head()`).
- **Table captions precede** the tabular; **figure captions follow** the graphic.
- **booktabs** rules, no vertical rules.
- **Figure widths** are given as a multiple of `\linewidth`, as the CFP asks.
- **Checklist** (`checklist.tex`) follows the references and the appendix. It is mandatory:
  papers without it are desk rejected. It does not count toward the page limit.
- **Fonts.** `analysis/fig_common.py` sets `pdf.fonttype = 42`, because matplotlib defaults
  to Type 3 and the CFP forbids Type 3 in the submitted PDF. Figures are vector PDF, not
  SVG or raster.

## Page limits

The attached formatting instructions are the **main track**: nine pages including figures.
The target venue is the **VLM4RWD workshop**, which is stricter at **eight pages**, excluding
references, checklist and appendix. `make_pdf.py` prints the body count against both.

## Two checklist answers that are "No" for a fixable reason

- **Compute resources.** The paper states token budgets but not accelerator type, memory or
  wall-clock. One sentence in Appendix F would turn this into a Yes.
- **Licenses for existing assets.** Every asset is cited, but no license is named. Listing
  the licenses for nanoVLM, SigLIP, SmolLM2, `the_cauldron` and FineVision would turn this
  into a Yes.

Both were left as honest `No` answers rather than being written into the paper, because the
conversion was scoped to not change content.

## Figures print at full width

`build_tex.py` defaults to `--figwidth=1.0`. After the September compaction pass the body sits
at 8 pages with both figures at the full 5.5in text width and captions at the style file's
10pt. Captions are *not* set in `\small`: the formatting instructions say "do not change font
sizes", so an earlier 9pt-caption experiment was reverted.

Figure legibility is solved at the source, not in LaTeX. `fig2_phase_portrait.py` and
`fig6_sink_stripe.py` now draw at 5.5in with 6-7.5pt fonts, so what matplotlib shows is what
prints. The appendix figures still need the same treatment; `docs/figure-plan.md` has the
list and the rule.

## Iterating on structure

Section order is the sort order of `sections/0N-*.md`, so reordering is a rename:

```
git mv sections/05-related-work.md sections/035-related-work.md
python3 check_refs.py
python3 make_pdf.py
```

The catch: section, figure, table and appendix numbers are **literal text** in the markdown.
Nothing resolves them, so a reorder silently invalidates references. There are about 83 of
them (40 `§N`, 8 figure, 10 table, 25 appendix). `check_refs.py` is the guard -- it derives
the real order from the files and reports every heading whose number no longer matches its
position, every `§N` pointing at nothing, figure and table captions numbered out of order,
and appendix sections the body never cites. Run it after every reorder, before rebuilding.

Tables with no caption are emitted inline rather than as floats, because an uncaptioned float
cannot be numbered and drifts to the top of a later page away from the sentence that
introduces it.
