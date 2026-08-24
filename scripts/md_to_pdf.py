"""Render a Markdown file to PDF using markdown-it-py and headless Edge.

No LaTeX, no pandoc, no extra install: markdown-it-py ships with rich, and Edge
is already on the machine. The Markdown becomes a styled HTML page, then Edge
prints that page to PDF with its own layout engine, so the inline SVG diagrams
and the shields.io badges render exactly as they do on GitHub.

Usage:
    python scripts/md_to_pdf.py                    # README.md -> README.pdf
    python scripts/md_to_pdf.py docs/notes.md      # -> docs/notes.pdf
    python scripts/md_to_pdf.py in.md out.pdf
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from markdown_it import MarkdownIt

EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")

# Print stylesheet. Sized for A4: tables and code blocks must never be split
# across a page break, and the widest code line (93 chars) has to fit in 178 mm.
CSS = """
@page { size: A4; margin: 18mm 15mm; }

body {
  font-family: 'Segoe UI', system-ui, sans-serif;
  font-size: 10.5pt;
  line-height: 1.55;
  color: #1E293B;
  max-width: 100%;
  margin: 0;
}

h1 { font-size: 22pt; font-weight: 700; margin: 0 0 4pt; letter-spacing: -0.4pt; }
h2 { font-size: 15pt; font-weight: 700; margin: 20pt 0 8pt; color: #111827; }
h3 { font-size: 12pt; font-weight: 600; margin: 14pt 0 6pt; color: #334155; }
h4 { font-size: 10.5pt; font-weight: 600; margin: 12pt 0 4pt; color: #475569; }
h1, h2, h3, h4 { break-after: avoid; }
/* A group heading must never be orphaned at the bottom of a page without its table. */
h3 + table, h4 + table, h3 + p, h4 + p { break-before: avoid; }

p { margin: 6pt 0; }
a { color: #1D4ED8; text-decoration: none; }

hr {
  border: none;
  border-top: 1px solid #E2E8F0;
  margin: 16pt 0;
}

/* Code: keep ASCII traces aligned, so no wrapping. 8pt keeps 93 chars on the page. */
pre {
  background: #F8FAFC;
  border: 1px solid #E2E8F0;
  border-radius: 6pt;
  padding: 8pt 10pt;
  font-family: 'Cascadia Mono', Consolas, monospace;
  font-size: 8pt;
  line-height: 1.45;
  white-space: pre;
  overflow: hidden;
  break-inside: avoid;
}
code {
  font-family: 'Cascadia Mono', Consolas, monospace;
  font-size: 9pt;
  background: #F1F5F9;
  padding: 1pt 3pt;
  border-radius: 3pt;
}
pre code { background: none; padding: 0; font-size: inherit; }

table {
  border-collapse: collapse;
  width: 100%;
  margin: 10pt 0;
  font-size: 9.5pt;
  break-inside: avoid;
}
th, td {
  border: 1px solid #E2E8F0;
  padding: 5pt 7pt;
  text-align: left;
  vertical-align: top;
}
th { background: #F8FAFC; font-weight: 600; }

blockquote {
  margin: 10pt 0;
  padding: 6pt 12pt;
  border-left: 3px solid #8B5CF6;
  background: #F5F3FF;
  break-inside: avoid;
}
blockquote p { margin: 0; }

img { max-width: 100%; height: auto; }
p[align="center"] { text-align: center; break-inside: avoid; }

/* <details> is collapsed by default in a browser, and a collapsed block prints
   as nothing. The renderer forces it open, so it only needs framing here. */
details {
  border: 1px solid #E2E8F0;
  border-radius: 6pt;
  padding: 8pt 12pt;
  margin: 10pt 0;
}
summary { font-weight: 600; margin-bottom: 6pt; }

li { margin: 3pt 0; }
"""

HTML_SHELL = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
{body}
</body>
</html>
"""


# Code blocks keep their alignment (white-space: pre), so a line wider than the
# printable area is silently clipped instead of wrapping. A4 minus the margins
# and the block padding leaves room for about 102 monospace characters at 8pt.
MAX_CODE_LINE = 100


def warn_on_wide_code_lines(text: str) -> None:
    """Report code lines that will be cut off on the right edge of the page."""
    offenders = []
    inside = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith("```"):
            inside = not inside
            continue
        if inside and len(line) > MAX_CODE_LINE:
            offenders.append((lineno, len(line), line.strip()[:60]))

    for lineno, width, preview in offenders:
        print(f"  ! line {lineno}: {width} chars, will be clipped -> {preview}...")
    if offenders:
        print(f"  {len(offenders)} line(s) over {MAX_CODE_LINE} chars, shorten them or they lose their tail")


def build_html(md_path: Path) -> str:
    """Convert Markdown to a standalone HTML page with absolute asset paths."""
    md = MarkdownIt("commonmark", {"html": True}).enable("table").enable("strikethrough")
    body = md.render(md_path.read_text(encoding="utf-8"))

    # Edge renders the HTML from the temp directory, so every relative asset
    # reference has to become an absolute file:// URL to survive the move.
    root = md_path.parent.resolve()
    for prefix in ('src="', 'href="'):
        for asset_dir in ("assets/", "img/", "docs/"):
            body = body.replace(f'{prefix}{asset_dir}', f'{prefix}{root.as_uri()}/{asset_dir}')

    # Collapsed on GitHub is the point, invisible in a printed PDF is not.
    body = body.replace("<details>", "<details open>")

    return HTML_SHELL.format(title=md_path.stem, css=CSS, body=body)


def print_to_pdf(html: str, pdf_path: Path) -> None:
    """Write the HTML to a temp file and let Edge print it."""
    if not EDGE.exists():
        raise SystemExit(f"Edge not found at {EDGE}")

    with tempfile.NamedTemporaryFile("w", suffix=".html", encoding="utf-8", delete=False) as f:
        f.write(html)
        html_path = Path(f.name)

    subprocess.run(
        [
            str(EDGE),
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            # Badges are fetched from shields.io, give the page time to load them.
            "--virtual-time-budget=20000",
            f"--print-to-pdf={pdf_path}",
            html_path.as_uri(),
        ],
        check=True,
        capture_output=True,
    )


def main() -> None:
    args = sys.argv[1:]
    md_path = Path(args[0]) if args else Path(__file__).resolve().parent.parent / "README.md"
    pdf_path = Path(args[1]) if len(args) > 1 else md_path.with_suffix(".pdf")

    if not md_path.exists():
        raise SystemExit(f"No such file: {md_path}")

    print(f"[1/3] reading   {md_path}")
    warn_on_wide_code_lines(md_path.read_text(encoding="utf-8"))
    html = build_html(md_path)

    print("[2/3] rendering  markdown -> html")
    print("[3/3] printing   html -> pdf (Edge headless, up to 20 s for the badges)")
    print_to_pdf(html, pdf_path.resolve())

    size_kb = pdf_path.stat().st_size / 1024
    print(f"\nDone: {pdf_path.resolve()}  ({size_kb:.0f} Ko)")


if __name__ == "__main__":
    main()
