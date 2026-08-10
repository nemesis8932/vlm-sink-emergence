#!/usr/bin/env python3
"""Assemble sections/*.md -> paper-v2-ste.html (+ draft-v2-ste.md), then print to PDF via Chrome.

STE edition: same content and figures as the VLM4RWD fork, prose rewritten in
ASD-STE100 Simplified Technical English (STE-flavored mode).

Usage: python3 build.py            # writes paper-v2-ste.html, draft-v2-ste.md, paper-v2-ste.pdf
Markdown subset: # h2 / ## h3, pipe tables, -/1. lists, **bold**, *italic*, `code`,
raw-HTML blocks (lines starting with '<' pass through), [n] refs styled when a block
starts with '['.
"""
import re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).parent
SECTIONS = sorted((HERE / "sections").glob("[0-9][0-9]-*.md"))

TITLE = "Attention-Sink Signatures Dissociate During Vision–Language Pretraining"

# Default = anonymous double-blind workshop build. `python3 build.py --arxiv` produces the
# named arXiv build (separate output files; the anonymous build is never overwritten).
ARXIV = "--arxiv" in sys.argv

if ARXIV:
    AUTHOR = "Samvat Tiwari"
    EMAIL = "Independent Researcher · samvat.t@gmail.com"
    DATE = ""                       # no draft/version/edition language in the arXiv build
    SLUG = "paper-v2-ste-arxiv"
else:
    AUTHOR = "Anonymous author(s)"    # double-blind: real name/email restored only in camera-ready
    EMAIL = "Submitted to VLM4RWD @ NeurIPS 2026"
    DATE = "July 2026 · Preprint draft v2 · Simplified Technical English edition"
    SLUG = "paper-v2-ste"

# The double-blind build withholds the repro links; the arXiv build shows them. Matched as a
# whitespace-tolerant regex because the source wraps this sentence across lines.
REPRO_ANON = re.compile(
    r"We\s+will\s+release\s+the\s+code,.*?double-blind\s+review\.", re.S)
REPRO_NAMED = ("We release the code, the probe, the run configurations, and the per-run logs at "
               "https://github.com/nemesis8932/vlm-sink-emergence, and the training checkpoints "
               "at https://huggingface.co/datasets/nemesismaniac/vlm-sink-emergence-ckpts.")

URL_RE = re.compile(r"(?<![\"'>=])(https?://[^\s<>)\]]+?)(?=[.,;:]?(?:\s|$|\)|\]))")
EQ_RE = re.compile(r"^(Sink<sup>|Sink\^|v-ratio\s*=|h-ratio\s*=)")

def linkify(s: str) -> str:
    """Bare URLs -> clickable anchors (arXiv readers get working repro links)."""
    return URL_RE.sub(r'<a href="\1">\1</a>', s)

def inline(s: str) -> str:
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*([^*\n]+?)\*(?![\w*])", r"<em>\1</em>", s)
    s = re.sub(r"`([^`]+?)`", r"<code>\1</code>", s)
    # Sink^0.2 / Sink^ε / Sink^ε_1 all carry the same "_1" subscript of the metric
    s = re.sub(r"Sink\^([0-9.]+|ε)(?:_1)?", r"Sink<sup>\1</sup><sub>1</sub>", s)
    # typographic quotes/apostrophes; the (?<!=) guard skips HTML attribute quotes
    s = re.sub(r'(?<!=)"([^"<>]*?)"', r"“\1”", s)
    s = re.sub(r"(\w)'(\w)", r"\1’\2", s)
    # Times carries ‖ but draws it as a thin single bar; force a math face for norm spans
    s = re.sub(r"‖([^‖]{1,60}?)‖", r'<span class="mth">‖\1‖</span>', s)
    s = linkify(s)
    return s

def table_html(lines):
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    head, body = rows[0], rows[2:]
    out = ["<table>", "<thead><tr>"]
    out += [f"<th>{inline(c)}</th>" for c in head]
    out.append("</tr></thead><tbody>")
    for r in body:
        out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)

def list_html(lines, ordered):
    items, cur = [], None
    pat = re.compile(r"^\s*\d+\.\s+" if ordered else r"^\s*-\s+")
    for ln in lines:
        if pat.match(ln):
            if cur is not None:
                items.append(cur)
            cur = pat.sub("", ln).strip()
        else:
            cur += " " + ln.strip()
    if cur is not None:
        items.append(cur)
    tag = "ol" if ordered else "ul"
    return f"<{tag}>" + "".join(f"<li>{inline(i)}</li>" for i in items) + f"</{tag}>"

def md_to_html(text: str) -> str:
    blocks, out = re.split(r"\n\s*\n", text), []
    for b in blocks:
        b = b.strip("\n")
        if not b.strip():
            continue
        lines = b.split("\n")
        first = lines[0].lstrip()
        if first.startswith("<"):
            out.append(b)
        elif len(lines) == 1 and EQ_RE.match(first):
            # displayed equation: centred, never justified or hyphenated
            out.append(f'<p class="eq">{inline(first)}</p>')
        elif first.startswith("## "):
            out.append(f"<h3>{inline(first[3:])}</h3>")
        elif first.startswith("# "):
            out.append(f"<h2>{inline(first[2:])}</h2>")
        elif first.startswith("|"):
            out.append(table_html(lines))
        elif re.match(r"^\s*-\s+", first):
            out.append(list_html(lines, ordered=False))
        elif re.match(r"^\s*\d+\.\s+", first):
            out.append(list_html(lines, ordered=True))
        else:
            para = inline(" ".join(ln.strip() for ln in lines))
            cls = ""
            if para.startswith("<strong>Table"):
                cls = ' class="table-caption"'
            elif para.startswith("["):
                cls = ' class="ref"'
            out.append(f"<p{cls}>{para}</p>")
    return "\n".join(out)

CSS = """
/* Layout skeleton follows the lean single-column arXiv/NeurIPS convention (cf.
   arXiv:2511.17036): letter page, ~1in side margins, Times-family serif at 10pt,
   bold run-in section heads, small justified captions. */
@page { size: letter; margin: 19mm 24mm 20mm 24mm; }
@page { @bottom-center { content: counter(page); font-size: 9pt; color: #444; } }
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; }
/* math fonts sit at the END of the stack: Latin text still renders in Times, and only the
   glyphs Times lacks (‖, ℓ, Σ) fall through to a math face */
body { font-family: 'Nimbus Roman', 'Times New Roman', 'Liberation Serif', Times,
                    'STIX Two Math', 'Cambria Math', serif;
       font-size: 10pt; line-height: 1.34; color: #111; margin: 0 auto; max-width: 6.5in;
       orphans: 3; widows: 3; }
a { color: #0b3d91; text-decoration: none; word-break: break-word; }
@media screen { body { padding: 40px 20px; } }
h1.title { font-size: 16.5pt; text-align: center; line-height: 1.25; margin: 0 0 0.7em; }
.author, .affil, .date { text-align: center; margin: 0.15em 0; }
.author { font-size: 11.5pt; }
.affil { font-size: 9.5pt; color: #444; }
.date { font-size: 9pt; color: #666; font-style: italic; margin-bottom: 1.6em; }
.abstract { margin: 0.4em 2.4em 1.6em; font-size: 9.6pt; line-height: 1.36;
            text-align: justify; hyphens: auto; }
.abstract p { margin: 0 0 0.5em; }
.abstract-head { font-weight: bold; font-variant: small-caps; letter-spacing: 0.03em;
                 margin-right: 0.15em; }
.mth { font-family: 'STIX Two Math', 'Cambria Math', 'Nimbus Roman', Times, serif; }
p.eq { text-align: center; hyphens: none; text-indent: 0;
       margin: 0.85em 0 0.95em; font-size: 10.5pt;
       font-family: 'STIX Two Math', 'Cambria Math', 'Nimbus Roman',
                    'Times New Roman', Times, serif; }
h2 { font-size: 12pt; font-weight: bold; margin: 1.45em 0 0.45em;
     break-after: avoid; page-break-after: avoid; }
h3 { font-size: 10.5pt; font-weight: bold; margin: 1.1em 0 0.35em;
     break-after: avoid; page-break-after: avoid; }
p { text-align: justify; hyphens: auto; margin: 0 0 0.5em; orphans: 3; widows: 3; }
ol, ul { margin: 0.2em 0 0.7em; padding-left: 1.7em; }
li { text-align: justify; margin-bottom: 0.35em; }
figure { margin: 1.2em 0 1.3em; break-inside: avoid; text-align: center; }
figure img { width: 100%; }
figcaption { font-size: 9pt; text-align: justify; color: #222; margin-top: 0.45em;
             line-height: 1.34; hyphens: auto; }
.table-caption { font-size: 9pt; margin: 0.9em 0 0.3em; }
table { margin: 0.3em auto 1em; border-collapse: collapse; font-size: 9pt;
        break-inside: avoid; }
thead tr { border-top: 1.4px solid #111; border-bottom: 0.7px solid #111; }
tbody tr:last-child { border-bottom: 1.4px solid #111; }
th, td { padding: 0.28em 0.65em; text-align: left; }
td:nth-child(n+3), th:nth-child(n+3) { text-align: center; }
code { font-family: 'Menlo', 'Consolas', monospace; font-size: 90%; }
.references p.ref { font-size: 9pt; padding-left: 1.7em; text-indent: -1.7em;
                    margin-bottom: 0.45em; text-align: left; }
.verify { color: #a11; font-size: 8pt; }
sup, sub { line-height: 0; }
"""

def section_text(p):
    t = p.read_text()
    if ARXIV:
        t = REPRO_ANON.sub(REPRO_NAMED, t)
    return t

def render_section(p):
    """Section -> HTML. The abstract gets the indented, bold-run-in-head block treatment."""
    txt = section_text(p)
    html = md_to_html(txt)
    if txt.lstrip().startswith("Abstract."):
        html = html.replace("<p>Abstract.",
                            '<p><span class="abstract-head">Abstract.</span>', 1)
        html = f'<div class="abstract">{html}</div>'
    return html


def build():
    body = "\n\n".join(render_section(p) for p in SECTIONS)
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{TITLE}</title><style>{CSS}</style></head>
<body>
<h1 class="title">{TITLE}</h1>
<p class="author">{AUTHOR}</p>
<p class="affil">{EMAIL}</p>
<p class="date">{DATE}</p>
{body}
</body></html>
"""
    (HERE / f"{SLUG}.html").write_text(html)
    stitched = "\n\n---\n\n".join(section_text(p) for p in SECTIONS)
    draft = "draft-v2-ste-arxiv.md" if ARXIV else "draft-v2-ste.md"
    (HERE / draft).write_text(f"# {TITLE}\n\n*{AUTHOR} — {DATE}*\n\n---\n\n{stitched}")
    print(f"wrote {SLUG}.html, {draft}")
    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    pdf = HERE / f"{SLUG}.pdf"
    subprocess.run([chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    "--virtual-time-budget=20000",
                    f"--print-to-pdf={pdf}",
                    (HERE / f"{SLUG}.html").as_uri()], check=True)
    print(f"wrote {SLUG}.pdf")
    stamp_metadata(pdf)


def stamp_metadata(pdf):
    """Chrome writes no Title/Author into the PDF. Stamp them with pypdf (an untitled PDF
    looks unfinished, and some readers surface this metadata). The anonymous build is
    stamped 'Anonymous' so the double-blind PDF never leaks the author through metadata."""
    title = TITLE.replace("–", "-").replace("—", "-")
    author = AUTHOR if ARXIV else "Anonymous author(s)"
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        print("[warn] PDF metadata not stamped (pypdf missing); pip install pypdf to enable")
        return
    reader = PdfReader(str(pdf))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.add_metadata({"/Title": title, "/Author": author,
                         "/Subject": "Attention sinks in vision-language pretraining",
                         "/Creator": "build.py"})
    with open(pdf, "wb") as f:
        writer.write(f)
    print(f"stamped metadata: Title='{title[:40]}...' Author='{author}'")

if __name__ == "__main__":
    build()
