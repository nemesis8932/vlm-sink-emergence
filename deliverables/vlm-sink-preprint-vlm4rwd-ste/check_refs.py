#!/usr/bin/env python3
"""Cross-reference linter for sections/*.md.

Section, figure, table and appendix numbers are literal text in the markdown -- nothing in
build_tex.py resolves them. So reordering sections (by renaming files) silently invalidates
every stale reference. Run this after any reorder, before rebuilding.

    python3 check_refs.py        exit 0 if clean, 1 if anything is stale
"""
import re
import sys
from pathlib import Path

SEC = Path(__file__).parent / "sections"
body = sorted(SEC.glob("0[2-7]-*.md"))
problems = []


def note(msg):
    problems.append(msg)


# --- what actually exists, in file order -------------------------------------------------
declared, sub = [], []
for i, f in enumerate(body, start=1):
    text = f.read_text()
    m = re.match(r"# (\d+)\. (.+)", text.splitlines()[0])
    if not m:
        note(f"{f.name}: first line is not '# N. Title'")
        continue
    declared.append((int(m.group(1)), f.name, m.group(2)))
    if int(m.group(1)) != i:
        note(f"{f.name}: heading says section {m.group(1)} but it is in position {i}. "
             f"Renumber the heading and every '§{m.group(1)}' that points at it.")
    for s in re.findall(r"^## (\d+)\.(\d+)", text, re.M):
        sub.append(f"{s[0]}.{s[1]}")
        if int(s[0]) != int(m.group(1)):
            note(f"{f.name}: subsection {s[0]}.{s[1]} sits under section {m.group(1)}")

top = {str(n) for n, _, _ in declared}
valid = top | set(sub)

# figures and tables, in order of appearance across the body
figs = [int(n) for f in body for n in re.findall(r"<b>\s*Figure\s+(\d+):", f.read_text())]
tabs = [int(n) for f in body for n in re.findall(r"^\*\*Table\s+(\d+)", f.read_text(), re.M)]
for name, got in (("Figure", figs), ("Table", tabs)):
    if got != list(range(1, len(got) + 1)):
        note(f"{name} captions appear in order {got}; LaTeX will number them "
             f"{list(range(1, len(got) + 1))}. Renumber the captions and their references.")

apx = set(re.findall(r"^## ([A-Z])\.", (SEC / "09-appendix.md").read_text(), re.M))

# --- what the prose claims ---------------------------------------------------------------
for f in body + [SEC / "09-appendix.md"]:
    text = f.read_text()
    for m in re.finditer(r"Sections? (\d+(?:\.\d+)?)(?: and (\d+(?:\.\d+)?))?", text):
        for ref in filter(None, m.groups()):
            if ref not in valid:
                note(f"{f.name}: Section {ref} has no such section")
    for kind, seen, ref in ([("Figure", figs, r) for r in
                             re.findall(r"(?:Fig\.|Figure) (\d+)", text)] +
                            [("Table", tabs, r) for r in re.findall(r"Table (\d+)", text)]):
        if int(ref) > len(seen):
            note(f"{f.name}: {kind} {ref} referenced but only {len(seen)} exist")
    for ref in set(re.findall(r"Appendix ([A-Z])", text)):
        if ref not in apx:
            note(f"{f.name}: Appendix {ref} has no such heading")

# --- appendix sections nothing points at --------------------------------------------------
allrefs = " ".join(f.read_text() for f in body)
for letter in sorted(apx):
    if not re.search(rf"Appendix {letter}\b", allrefs):
        note(f"Appendix {letter} is never referenced from the body")

print(f"body order: " + " -> ".join(f"§{n} {t}" for n, _, t in declared))
print(f"{len(figs)} figures, {len(tabs)} tables, {len(apx)} appendix sections")
if problems:
    print(f"\n{len(problems)} problem(s):")
    for p in problems:
        print("  -", p)
    sys.exit(1)
print("\ncross-references clean")
