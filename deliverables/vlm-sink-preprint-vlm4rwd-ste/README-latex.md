# NeurIPS 2026 LaTeX build

The markdown in `sections/` stays the single source. Two backends read it:

| backend | driver | output | purpose |
|---|---|---|---|
| HTML/PDF reading copy | `build.py` | `paper-v2-ste.pdf` | fast to read, not a submission format |
| NeurIPS LaTeX | `build_tex.py` → `make_pdf.py` | `paper-neurips.pdf` | the actual submission |

Edit prose only in `sections/*.md`. Never hand-edit the generated `.tex`.

## Two steps I could not do for you

Both are blocked inside the agent sandbox (no sudo, no network egress).

**1. Install pdflatex.** The call for papers says to generate the PDF with `pdflatex`.

```
brew install --cask basictex          # ~100 MB; prompts for your password
eval "$(/usr/libexec/path_helper)"    # or open a new shell
sudo tlmgr update --self
sudo tlmgr install type1cm cm-super microtype booktabs natbib nicefrac
```

**2. Get the official style file.** `neurips_2026.sty` must sit in this directory, and must
be the unmodified file from the conference — "Tweaking the style files may be grounds for
desk rejection." The 2026 Styles archive was not resolvable from here
(`media.neurips.cc/Conferences/NeurIPS2026/Styles.zip` returns 404), so fetch it from the
Paper Information pages at <https://neurips.cc>, or from the official Overleaf template.

Then:

```
python3 make_pdf.py              # anonymous submission build, with line numbers
python3 make_pdf.py --preprint   # named arXiv build
```

`make_pdf.py` reports total pages, body pages, and whether any Type 3 font slipped in.

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
