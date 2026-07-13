#!/usr/bin/env python3
"""Assemble sections/*.md -> paper-v2.html (+ draft-v2.md), then print to PDF via Chrome.

Usage: python3 build.py            # writes paper-v2.html, draft-v2.md, paper-v2.pdf
Markdown subset: # h2 / ## h3, pipe tables, -/1. lists, **bold**, *italic*, `code`,
raw-HTML blocks (lines starting with '<' pass through), [n] refs styled when a block
starts with '['.
"""
import re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).parent
SECTIONS = sorted((HERE / "sections").glob("[0-9][0-9]-*.md"))

TITLE = "Four Levers, Four Corners: Attention-Sink Signatures Dissociate in From-Scratch Vision–Language Pretraining — A Precondition for Grounding Fidelity"
AUTHOR = "Anonymous author(s)"    # double-blind: real name/email restored only in camera-ready
EMAIL = "Submitted to VLM4RWD @ NeurIPS 2026"
DATE = "July 2026 · Preprint draft v2"

def inline(s: str) -> str:
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*([^*\n]+?)\*(?![\w*])", r"<em>\1</em>", s)
    s = re.sub(r"`([^`]+?)`", r"<code>\1</code>", s)
    s = re.sub(r"Sink\^([0-9.]+|ε)", r"Sink<sup>\1</sup>", s)
    # typographic quotes/apostrophes; the (?<!=) guard skips HTML attribute quotes
    s = re.sub(r'(?<!=)"([^"<>]*?)"', r"“\1”", s)
    s = re.sub(r"(\w)'(\w)", r"\1’\2", s)
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
@page { size: letter; margin: 20mm 18mm 22mm 18mm; }
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; }
body { font-family: 'Charter', 'Bitstream Charter', 'Iowan Old Style', 'Georgia', serif;
       font-size: 10.5pt; line-height: 1.42; color: #111; margin: 0 auto; max-width: 7.3in; }
@media screen { body { padding: 40px 20px; } }
h1.title { font-size: 16.5pt; text-align: center; line-height: 1.25; margin: 0 0 0.7em; }
.author, .affil, .date { text-align: center; margin: 0.15em 0; }
.author { font-size: 11.5pt; }
.affil { font-size: 9.5pt; color: #444; }
.date { font-size: 9pt; color: #666; font-style: italic; margin-bottom: 1.6em; }
.abstract { margin: 0 2.1em 1.4em; font-size: 9.8pt; text-align: justify; hyphens: auto; }
.abstract-head { font-weight: bold; }
h2 { font-size: 12.5pt; margin: 1.5em 0 0.5em; break-after: avoid; }
h3 { font-size: 11pt; margin: 1.2em 0 0.4em; break-after: avoid; }
p { text-align: justify; hyphens: auto; margin: 0 0 0.55em; }
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

def build():
    body = "\n\n".join(md_to_html(p.read_text()) for p in SECTIONS)
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
    (HERE / "paper-v2.html").write_text(html)
    stitched = "\n\n---\n\n".join(p.read_text() for p in SECTIONS)
    (HERE / "draft-v2.md").write_text(f"# {TITLE}\n\n*{AUTHOR} — {DATE}*\n\n---\n\n{stitched}")
    print("wrote paper-v2.html, draft-v2.md")
    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    subprocess.run([chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    "--virtual-time-budget=20000",
                    f"--print-to-pdf={HERE / 'paper-v2.pdf'}",
                    (HERE / "paper-v2.html").as_uri()], check=True)
    print("wrote paper-v2.pdf")

if __name__ == "__main__":
    build()
