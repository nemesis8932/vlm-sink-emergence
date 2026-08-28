#!/usr/bin/env python3
"""Assemble sections/*.md -> paper.tex for the official NeurIPS 2026 template.

The markdown in sections/ stays canonical; this is a second backend alongside build.py
(which produces the HTML/PDF reading copy). Nothing here edits prose.

Usage:
    python3 build_tex.py              # anonymous submission build (line numbers, no author)
    python3 build_tex.py --preprint   # arXiv build (named, "Preprint. Work in progress.")

Conventions this enforces, from the NeurIPS 2026 formatting instructions:
  * headings lower case except first word and proper nouns
  * table caption BEFORE the tabular, figure caption AFTER the graphic
  * booktabs rules, no vertical rules
  * \\includegraphics width given as a multiple of \\linewidth
  * abstract is a single paragraph
  * the paper checklist follows the references and the appendix
"""
import html
import re, sys
from pathlib import Path

HERE = Path(__file__).parent
SEC = HERE / "sections"
PREPRINT = "--preprint" in sys.argv

TITLE = "Attention-Sink Signatures Dissociate During Vision--Language Pretraining"

# ---------------------------------------------------------------- inline conversion

UNICODE = {
    "\u00a7": r"\S",        "\u2013": "--",         "\u2014": "---",
    "\u2192": r"$\rightarrow$", "\u03b5": r"$\epsilon$", "\u2212": "$-$",
    "\u00d7": r"$\times$",  "\u03c3": r"$\sigma$",  "\u00b7": r"$\cdot$",
    "\u2248": r"$\approx$", "\u2016": r"$\|$",      "\u2208": r"$\in$",
    "\u00e7": r"\c{c}",     "\u201c": "``",         "\u201d": "''",
    "\u2019": "'",          "\u2264": r"$\leq$",    "\u2265": r"$\geq$",
}
SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
           "<": "$<$", ">": "$>$",
           "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
           "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def _protect(s, store, render):
    """Pull a construct out of the text so escaping cannot touch it."""
    store.append(render)
    return f"\x00{len(store)-1}\x00"


def inline(s, cite=True):
    held = []

    # HTML entities: decode them before anything escapes the bare "&". Hold back the
    # angle-bracket ones first so they cannot be mistaken for tags in the next step.
    s = s.replace("&lt;", "\x03").replace("&gt;", "\x04")
    s = html.unescape(s)

    # HTML inline emphasis -> markdown, so a single code path produces the braces
    s = re.sub(r"<(?:em|i)>(.*?)</(?:em|i)>", r"*\1*", s, flags=re.S)
    s = re.sub(r"<(?:strong|b)>(.*?)</(?:strong|b)>", r"**\1**", s, flags=re.S)

    # citations first: after escaping, an en-dash range has become "--" and stops matching
    if cite:
        s = CITE.sub(lambda m: _protect(s, held, _cite(m)), s)

    # code spans survive verbatim inside \texttt
    def code(m):
        inner = m.group(1)
        for ch, rep in SPECIAL.items():
            inner = inner.replace(ch, rep)
        return _protect(s, held, r"\texttt{" + inner + "}")
    s = re.sub(r"`([^`]+?)`", code, s)

    s = re.sub(r"<sup>(.*?)</sup>", lambda m: _protect(s, held, "$^{" + _math(m.group(1)) + "}$"), s)
    s = re.sub(r"<sub>(.*?)</sub>", lambda m: _protect(s, held, "$_{" + _math(m.group(1)) + "}$"), s)

    # straight double quotes -> TeX quotes, in pairs
    s = re.sub(r'"([^"]*)"', r"``\1''", s)

    # Sink^0.2 / Sink^eps / Sink^eps_1 -> one math atom
    s = re.sub(r"Sink\^([0-9.]+|\u03b5)(?:_1)?",
               lambda m: _protect(s, held, r"$\mathrm{Sink}^{" + _math(m.group(1)) + r"}_{1}$"), s)

    # escape everything left
    out = []
    for ch in s:
        out.append(SPECIAL.get(ch, UNICODE.get(ch, ch)))
    s = "".join(out)

    # markdown emphasis (after escaping, so the markers are still bare)
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<![\w*])\*([^*\n]+?)\*(?![\w*])", r"\\textit{\1}", s)

    s = s.replace("\x03", "$<$").replace("\x04", "$>$")
    for i, r in enumerate(held):
        s = s.replace(f"\x00{i}\x00", r)
    return s


def _math(x):
    return x.replace("\u03b5", r"\epsilon").replace("\u2212", "-")


def _balance(s):
    """<em>x</em> became '\\textit{x\\textit{'; turn every second marker into a brace."""
    for marker, cmd in (("\x01", r"\textit{"), ("\x02", r"\textbf{")):
        parts = s.split(cmd)
        if len(parts) > 1:
            rebuilt, open_ = parts[0], False
            for p in parts[1:]:
                rebuilt += ("}" if open_ else cmd) + p
                open_ = not open_
            s = rebuilt
    return s


CITE = re.compile(r"\[(\d+(?:\s*[,\u2013-]\s*\d+)*)\]")


def _cite(m):
    keys = []
    for part in m.group(1).split(","):
        part = part.strip()
        if re.fullmatch(r"\d+", part):
            keys.append(int(part))
        else:
            a, b = re.split(r"[\u2013-]", part)
            keys.extend(range(int(a), int(b) + 1))
    return r"\cite{" + ",".join(f"ref{k}" for k in keys) + "}"


def lower_head(t):
    """CFP: headings lower case except the first word and proper nouns."""
    PROPER = {"RF", "MMStar", "SigLIP", "SmolLM2", "Sink", "KV", "AdamW", "FineVision"}
    words = t.split()
    out = [words[0]] if words else []
    for w in words[1:]:
        core = w.strip("(),.")
        out.append(w if (core in PROPER or not core[:1].isupper()) else w[0].lower() + w[1:])
    return " ".join(out)

# ---------------------------------------------------------------- block conversion

FIG_RE = re.compile(r'<figure id="([^"]+)">.*?<img src="([^"]+)".*?<figcaption>(.*?)</figcaption>.*?</figure>', re.S)


def figure_block(b, width):
    m = FIG_RE.search(b)
    if not m:
        raise SystemExit("unparsed figure block:\n" + b[:200])
    fid, src, cap = m.group(1), m.group(2), m.group(3)
    src = src.replace(".svg", ".pdf")            # pdflatex needs the vector PDF
    cap = " ".join(cap.split())
    # \caption already prints "Figure N:", so strip the number we carry in the source
    cap = re.sub(r"^<b>\s*Figure\s+[A-Z]?\d+:\s*", "<b>", cap)
    return ("\\begin{figure}[t]\n  \\centering\n"
            f"  \\includegraphics[width={width}\\linewidth]{{{src}}}\n"
            f"  \\caption{{{inline(cap)}}}\n  \\label{{fig:{fid}}}\n"
            "\\end{figure}")


def table_block(lines, caption):
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    head, body = rows[0], rows[2:]
    ncol = len(head)
    # centre a column only when every cell in it is short; long prose stays left
    align = []
    for j in range(ncol):
        cells = [r[j] for r in body if j < len(r)]
        align.append("l" if j == 0 or any(len(c) > 14 for c in cells) else "c")
    # a column holding prose cannot be an "l" column: it will not wrap and overflows the
    # 5.5in text block. Give those X columns and let tabularx share the leftover width.
    for j in range(ncol):
        cells = [r[j] for r in body if j < len(r)] + [head[j]]
        if any(len(c) > 24 for c in cells):
            align[j] = "X"
    env = "tabularx" if "X" in align else "tabular"
    spec = ("{\\linewidth}{" + "".join(align) + "}") if env == "tabularx" else ("{" + "".join(align) + "}")

    out = ["\\begin{table}[t]", "  \\centering", "  \\small"]
    if caption:
        out.append(f"  \\caption{{{inline(caption)}}}")     # CFP: caption BEFORE the table
    out.append(f"  \\begin{{{env}}}" + spec)
    out.append("    \\toprule")
    out.append("    " + " & ".join(inline(c) for c in head) + " \\\\")
    out.append("    \\midrule")
    for r in body:
        out.append("    " + " & ".join(inline(c) for c in r) + " \\\\")
    out.append("    \\bottomrule")
    out.append(f"  \\end{{{env}}}")
    out.append("\\end{table}")
    return "\n".join(out)


RUNIN = re.compile(r"^\*\*([^*]+?)\.?\*\*\s+(.*)$", re.S)
TABCAP = re.compile(r"^\*\*Table\s")


def convert(text, fig_width, top="section"):
    """Markdown -> LaTeX for one section file."""
    blocks = re.split(r"\n\s*\n", text)
    out, pending_caption = [], None
    depth = {"section": 0, "subsection": 1, "subsubsection": 2}[top]
    levels = ["section", "subsection", "subsubsection", "paragraph"]

    for b in blocks:
        b = b.strip("\n")
        if not b.strip():
            continue
        lines = b.split("\n")
        first = lines[0].lstrip()

        if first.startswith("<figure"):
            out.append(figure_block(b, fig_width)); continue
        if first.startswith("|"):
            out.append(table_block(lines, pending_caption)); pending_caption = None; continue
        if TABCAP.match(first):
            raw = " ".join(l.strip() for l in lines)
            # "**Table 1 - short title.** rest of the caption"; \caption supplies "Table N:"
            m = re.match(r"^\*\*Table\s+\d+\s*[\u2014\u2013-]?\s*(.*?)\*\*\s*(.*)$", raw, re.S)
            if m:
                title = m.group(1).strip()
                title = title[:1].upper() + title[1:]      # CFP: first word capitalised
                pending_caption = f"**{title}** {m.group(2).strip()}".strip()
            else:
                pending_caption = re.sub(r"\*\*", "", raw)
            continue
        for marks, lvl in (("### ", 2), ("## ", 1), ("# ", 0)):
            if first.startswith(marks):
                title = first[len(marks):].strip()
                title = re.sub(r"^(?:\d+(?:\.\d+)*\.?|[A-H](?:\.\d+)*\.?)\s+", "", title)
                out.append(f"\\{levels[min(depth+lvl,3)]}{{{inline(lower_head(title), cite=False)}}}")
                break
        else:
            if re.match(r"^\s*\d+\.\s+", first):
                items = _items(lines, r"^\s*\d+\.\s+")
                out.append("\\begin{enumerate}\n" + "".join(f"  \\item {inline(i)}\n" for i in items) + "\\end{enumerate}")
            elif re.match(r"^\s*-\s+", first):
                items = _items(lines, r"^\s*-\s+")
                out.append("\\begin{itemize}\n" + "".join(f"  \\item {inline(i)}\n" for i in items) + "\\end{itemize}")
            else:
                para = " ".join(l.strip() for l in lines)
                m = RUNIN.match(para)
                if m and not para.startswith("**Table"):
                    out.append(f"\\paragraph{{{inline(m.group(1), cite=False)}}} {inline(m.group(2))}")
                else:
                    out.append(inline(para))
    return "\n\n".join(out)


def _items(lines, pat):
    items, cur = [], None
    rx = re.compile(pat)
    for ln in lines:
        if rx.match(ln):
            if cur is not None:
                items.append(cur)
            cur = rx.sub("", ln).strip()
        else:
            cur += " " + ln.strip()
    if cur is not None:
        items.append(cur)
    return items

# ---------------------------------------------------------------- document assembly

PREAMBLE = r"""\documentclass{article}

%% The style file loads natbib for us, defaulting to author-year. Our bibliography is a
%% hand-built numeric list, so ask natbib for numeric citations first. The call for papers
%% documents exactly this route: "If you wish to load the natbib package with options, you
%% may add the following before loading the neurips_2026 package."
\PassOptionsToPackage{numbers}{natbib}

%% NeurIPS 2026. Submission build takes no option: the style file anonymises the paper and
%% adds line numbers, as the call for papers requires. --preprint passes [preprint] for the
%% arXiv copy. [final] is only for accepted papers and is never emitted here.
\usepackage%(opt)s{neurips_2026}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{url}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{amsfonts}
\usepackage{amsmath}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{hyperref}

\title{%(title)s}

\author{%(author)s}

\begin{document}

\maketitle
"""

AUTHOR_NAMED = r"""  Samvat Tiwari \\
  Independent Researcher \\
  \texttt{samvat.t@gmail.com} \\
"""
AUTHOR_ANON = r"""  Anonymous Author(s) \\
  Affiliation \\
  Address \\
  \texttt{email} \\
"""


def references(text):
    entries = []
    for b in re.split(r"\n\s*\n", text):
        b = b.strip()
        if not b.startswith("["):
            continue
        m = re.match(r"\[(\d+)\]\s*(.*)", " ".join(l.strip() for l in b.split("\n")), re.S)
        entries.append((int(m.group(1)), inline(m.group(2), cite=False)))
    entries.sort()
    body = "\n\n".join(f"\\bibitem{{ref{n}}}\n{txt}" for n, txt in entries)
    return ("\\small\n\\begin{thebibliography}{%d}\n\n" % len(entries)) + body + "\n\n\\end{thebibliography}\n\\normalsize"


def main():
    # 0.68 is the measured ceiling: the body lands on exactly 8 pages, the VLM4RWD limit.
    # 0.72 and above spill the conclusion onto a 9th page. Override with --figwidth=.
    fig_width = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--figwidth=")), "0.68")
    opt = "[preprint]" if PREPRINT else ""
    tex = [PREAMBLE % dict(opt=opt, title=TITLE,
                           author=AUTHOR_NAMED if PREPRINT else AUTHOR_ANON)]

    abstract = (SEC / "01-abstract.md").read_text().strip()
    abstract = re.sub(r"^Abstract\.\s*", "", abstract)
    abstract = " ".join(abstract.split())          # CFP: one paragraph only
    tex.append("\\begin{abstract}\n" + inline(abstract) + "\n\\end{abstract}\n")

    for f in sorted(SEC.glob("0[2-7]-*.md")):
        tex.append(convert(f.read_text(), fig_width))

    tex.append(references((SEC / "08-references.md").read_text()))

    tex.append("\n\\newpage\n\\appendix")
    app = (SEC / "09-appendix.md").read_text()
    app = re.sub(r"^# Appendix\s*\n", "", app)      # \appendix already supplies the part title
    app = re.sub(r"^## ", "# ", app, flags=re.M)     # promote A/B/C... to \section -> A, B, C
    app = re.sub(r"^### ", "## ", app, flags=re.M)
    tex.append(convert(app, fig_width))

    checklist = HERE / "checklist.tex"
    if checklist.exists():
        tex.append("\n\\newpage\n\\input{checklist}")

    tex.append("\n\\end{document}")
    out = HERE / ("paper-neurips-preprint.tex" if PREPRINT else "paper-neurips.tex")
    out.write_text("\n\n".join(tex) + "\n")
    print(f"wrote {out.name}  ({len(out.read_text().splitlines())} lines, "
          f"{'preprint/named' if PREPRINT else 'submission/anonymous'})")


if __name__ == "__main__":
    main()
