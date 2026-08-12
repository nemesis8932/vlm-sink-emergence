#!/usr/bin/env python3
"""Measure the BODY page count (sections 01-07 only) under NeurIPS 2026 page geometry.

The ≤8pp constraint is on the body, excluding references and appendix, so this renders
only sections 01–07 and counts pages.

Geometry follows neurips_2024.sty: single column, 5.5in x 9in text block on US Letter,
10pt Times, \\parskip between paragraphs. This is a geometric proxy, not a LaTeX render —
Chrome's Times metrics and line-breaking differ slightly from pdfTeX's, so treat the
number as ±0.3pp, and treat "8.0" as "at the line", not "safe".

Usage: python3 measure_pages.py
"""
import re, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build import md_to_html, TITLE          # reuse the same markdown subset

HERE = Path(__file__).parent
BODY = sorted((HERE / "sections").glob("0[1-7]-*.md"))

# neurips_2024.sty: \textwidth 5.5in, \textheight 9in, 10pt, Times.
CSS = """
@page { size: letter; margin: 1in 1.5in 1in 1.5in; }
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; }
body { font-family: 'Times New Roman', Times, serif; font-size: 10pt; line-height: 1.18;
       color: #000; margin: 0; width: 5.5in; }
h1.title { font-size: 14pt; font-weight: bold; text-align: center; line-height: 1.15;
           margin: 0 0 0.55em; }
.author, .affil, .date { text-align: center; margin: 0.1em 0; font-size: 10pt; }
.date { margin-bottom: 1.1em; }
.abstract { margin: 0 0.5in 1.1em; font-size: 10pt; text-align: justify; hyphens: auto; }
.abstract-head { font-weight: bold; }
h2 { font-size: 12pt; margin: 1.1em 0 0.4em; break-after: avoid; }
h3 { font-size: 10pt; font-weight: bold; margin: 0.9em 0 0.3em; break-after: avoid; }
h4 { font-size: 10pt; font-style: italic; margin: 0.7em 0 0.25em; break-after: avoid; }
p  { text-align: justify; hyphens: auto; margin: 0 0 0.5em; }
p.eq { text-align: center; hyphens: none; }
ol, ul { margin: 0.15em 0 0.55em; padding-left: 1.5em; }
li { text-align: justify; margin-bottom: 0.25em; }
figure { margin: 0.9em 0 1em; break-inside: avoid; text-align: center; }
/* 80%: the body fits 8pp at this width (measured); 100% spills to 9 */
figure img { width: 80%; }
figcaption { font-size: 9pt; text-align: justify; color: #000; margin-top: 0.35em;
             line-height: 1.15; hyphens: auto; }
.table-caption { font-size: 9pt; margin: 0.7em 0 0.25em; }
table { margin: 0.25em auto 0.8em; border-collapse: collapse; font-size: 9pt;
        break-inside: avoid; }
thead tr { border-top: 1.1px solid #000; border-bottom: 0.6px solid #000; }
tbody tr:last-child { border-bottom: 1.1px solid #000; }
th, td { padding: 0.2em 0.5em; text-align: left; }
td:nth-child(n+3), th:nth-child(n+3) { text-align: center; }
code { font-family: 'Courier New', monospace; font-size: 90%; }
.mth { font-family: 'STIXGeneral', 'Cambria Math', serif; }
a.cite { color: inherit; text-decoration: none; }
sup, sub { line-height: 0; }
"""

def main():
    body = "\n\n".join(md_to_html(p.read_text()) for p in BODY)
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{TITLE}</title><style>{CSS}</style></head>
<body>
<h1 class="title">{TITLE}</h1>
<p class="author">Anonymous author(s)</p>
<p class="date">&nbsp;</p>
{body}
</body></html>
"""
    out_html = HERE / "_bodycount.html"
    out_pdf = HERE / "_bodycount.pdf"
    out_html.write_text(html)
    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    subprocess.run([chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    "--virtual-time-budget=20000", f"--print-to-pdf={out_pdf}",
                    out_html.as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pages = out_pdf.read_bytes().count(b"/Type /Page\n") or \
            out_pdf.read_bytes().count(b"/Type/Page")
    words = sum(len(re.sub(r"<[^>]+>", " ", p.read_text()).split()) for p in BODY)
    print(f"BODY (§1–7) = {pages} pages, {words} words   [NeurIPS geometry proxy, ±0.3pp]")
    print(f"   limit 8pp -> {'OK' if pages <= 8 else 'OVER by ' + str(pages - 8)}")
    for p in BODY:
        print(f"   {p.name:20} {len(re.sub(r'<[^>]+>',' ',p.read_text()).split()):5} words")

if __name__ == "__main__":
    main()
