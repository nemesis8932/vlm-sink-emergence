#!/usr/bin/env python3
"""Build the NeurIPS 2026 PDF from sections/*.md and report the compliance facts.

    python3 make_pdf.py              submission build (anonymous, line-numbered)
    python3 make_pdf.py --preprint   arXiv build (named, "Preprint. Work in progress.")

Requires pdflatex on PATH and the official neurips_2026.sty beside this file. The style
file must be the unmodified one from neurips.cc -- the call for papers says tweaking it may
be grounds for desk rejection.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

# BasicTeX installs to /Library/TeX/texbin and only reaches PATH after a new login shell,
# so add it here rather than making the caller restart their terminal.
if not shutil.which("pdflatex") and Path("/Library/TeX/texbin/pdflatex").exists():
    os.environ["PATH"] = "/Library/TeX/texbin:" + os.environ.get("PATH", "")

HERE = Path(__file__).parent
PREPRINT = "--preprint" in sys.argv
SMOKE = "--smoketest" in sys.argv
SLUG = "paper-neurips-preprint" if PREPRINT else "paper-neurips"


def die(msg):
    sys.exit(f"ERROR: {msg}")


def smoketest():
    """Compile the generated LaTeX against plain article, with the style file's macros
    stubbed out. Proves the conversion is valid LaTeX when neurips_2026.sty is not yet
    available: a failure here is our bug, not a missing style file. Page geometry is NOT
    NeurIPS geometry, so the page count from this build means nothing."""
    subprocess.run([sys.executable, "build_tex.py"], cwd=HERE, check=True)
    src = (HERE / "paper-neurips.tex").read_text().replace(
        "\\usepackage{neurips_2026}",
        "\\usepackage[letterpaper,margin=1in]{geometry}\n"
        "\\newcommand{\\answerYes}[1][]{Yes}\n"
        "\\newcommand{\\answerNo}[1][]{No}\n"
        "\\newcommand{\\answerNA}[1][]{NA}")
    (HERE / "_smoketest.tex").write_text(src)
    log = ""
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-file-line-error",
                            "_smoketest.tex"], cwd=HERE, capture_output=True, text=True)
        log = r.stdout
    errs = [l for l in log.splitlines() if l.startswith("!")]
    over = [l for l in log.splitlines() if "Overfull" in l]
    undef = [l for l in log.splitlines() if "undefined" in l.lower() and "citation" in l.lower()]
    print("-" * 62)
    print(f"smoke test (plain article, NOT NeurIPS geometry)")
    print(f"  LaTeX errors      : {len(errs)}")
    for e in errs[:8]:
        print("     ", e)
    print(f"  overfull boxes    : {len(over)}")
    print(f"  undefined cites   : {len(undef)}")
    print("-" * 62)
    return 1 if errs else 0


def main():
    if not shutil.which("pdflatex"):
        die("pdflatex not found. Install BasicTeX or MacTeX, then re-run.")
    if SMOKE:
        return smoketest()
    if not (HERE / "neurips_2026.sty").exists():
        die("neurips_2026.sty missing. Download the official Styles archive from "
            "neurips.cc and put neurips_2026.sty in this directory.\n"
            "       Run with --smoketest to syntax-check the generated LaTeX without it.")

    subprocess.run([sys.executable, "build_tex.py"] + (["--preprint"] if PREPRINT else []),
                   cwd=HERE, check=True)

    # two passes: \cite numbering and float placement both settle on the second
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", f"{SLUG}.tex"],
                           cwd=HERE, capture_output=True, text=True)
        if r.returncode != 0:
            tail = [l for l in r.stdout.splitlines() if l.startswith("!")][:10]
            die("pdflatex failed:\n  " + "\n  ".join(tail or r.stdout.splitlines()[-15:]))

    pdf = HERE / f"{SLUG}.pdf"
    txt = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True).stdout
    pages = txt.split("\f")
    body = next((i for i, p in enumerate(pages)
                 if p.lstrip().startswith("References") or "\nReferences\n" in p), len(pages))

    fonts = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True).stdout
    kinds = sorted({" ".join(l.split()[1:3]) for l in fonts.splitlines()[2:] if l.split()})
    type3 = [k for k in kinds if "Type 3" in k]

    print("-" * 62)
    print(f"{pdf.name}: {len(pages) - 1} pages total")
    print(f"  body through the conclusion = {body} pages"
          "   [VLM4RWD workshop limit 8; NeurIPS main track 9]")
    print(f"  font types: {', '.join(kinds) or 'none'}")
    print("  Type 3 fonts:", "NONE (compliant)" if not type3 else f"VIOLATION {type3}")
    print("-" * 62)


if __name__ == "__main__":
    main()
