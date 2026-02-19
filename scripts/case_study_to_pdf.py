#!/usr/bin/env python3
"""
Convert a case study markdown file to a styled PDF.

Usage:
    python scripts/case_study_to_pdf.py <case_study_dir>
    python scripts/case_study_to_pdf.py output/case-studies/world-action-models

The script will:
  1. Read case_study.md from the directory
  2. Render LaTeX math as images via matplotlib
  3. Convert markdown tables, code blocks, and text to styled HTML
  4. Render a professional PDF
  5. Save it to the same directory as case_study.pdf
"""

import argparse
import hashlib
import re
import sys
import tempfile
from pathlib import Path

VENV_SITE = Path(__file__).resolve().parent.parent / ".venv/lib"
for p in sorted(VENV_SITE.glob("python*/site-packages")):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import markdown
from weasyprint import HTML

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSS = """
@page {
    size: A4;
    margin: 1.8cm 2.0cm 1.8cm 2.0cm;
    @bottom-center {
        content: "Vizuara AI — Case Study";
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 9pt;
        color: #999;
    }
    @bottom-right {
        content: counter(page);
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 9pt;
        color: #999;
    }
}

body {
    font-family: 'Georgia', 'Times New Roman', serif;
    font-size: 11pt;
    line-height: 1.5;
    color: #1a1a1a;
    max-width: 100%;
}

h1 {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 24pt;
    font-weight: 700;
    color: #111;
    margin-top: 0;
    margin-bottom: 0.2em;
    line-height: 1.2;
}

h1 + h2:first-of-type {
    font-size: 13pt;
    color: #555;
    font-style: italic;
    margin-bottom: 1em;
    border-bottom: 2px solid #e0e0e0;
    padding-bottom: 0.6em;
    font-weight: 400;
}

h2 {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 17pt;
    font-weight: 700;
    color: #222;
    margin-top: 1.2em;
    margin-bottom: 0.4em;
    border-bottom: 1px solid #e8e8e8;
    padding-bottom: 0.2em;
}

h3 {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 13pt;
    font-weight: 600;
    color: #333;
    margin-top: 1.0em;
    margin-bottom: 0.3em;
}

h4 {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 11.5pt;
    font-weight: 600;
    color: #444;
    margin-top: 0.8em;
    margin-bottom: 0.2em;
}

p {
    margin-bottom: 0.5em;
    text-align: left;
}

strong { font-weight: 700; }
em { font-style: italic; }

/* Tables */
table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.6em 0;
    font-size: 9.5pt;
}

th {
    background: #f0f2f5;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-weight: 600;
    text-align: left;
    padding: 0.5em 0.7em;
    border: 1px solid #d0d0d0;
}

td {
    padding: 0.4em 0.7em;
    border: 1px solid #d0d0d0;
    vertical-align: top;
}

tr:nth-child(even) td {
    background: #fafafa;
}

/* Code blocks */
pre {
    background: #f6f8fa;
    border: 1px solid #e1e4e8;
    border-radius: 6px;
    padding: 0.8em;
    font-size: 8.5pt;
    line-height: 1.45;
    overflow-x: auto;
    margin: 0.5em 0;
}

code {
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 8.5pt;
    background: #f0f2f4;
    padding: 0.1em 0.3em;
    border-radius: 3px;
}

pre code {
    background: none;
    padding: 0;
    border-radius: 0;
    font-size: inherit;
}

/* Block math */
.math-block {
    text-align: center;
    margin: 0.4em 0;
}
.math-block img {
    display: inline-block;
    height: auto;
    vertical-align: middle;
}

/* Inline math */
.math-inline img {
    display: inline;
    height: 1.2em;
    vertical-align: -0.3em;
    margin: 0 0.05em;
}

blockquote {
    border-left: 4px solid #4a90d9;
    margin: 0.6em 0;
    padding: 0.4em 1em;
    background: #f8f9fb;
    color: #444;
}

hr {
    border: none;
    border-top: 1px solid #e0e0e0;
    margin: 1.5em 0;
}

ol, ul {
    margin-bottom: 0.5em;
    padding-left: 1.5em;
}

li { margin-bottom: 0.25em; }

/* ASCII art diagrams */
pre:has(code) {
    font-size: 7.5pt;
    line-height: 1.3;
}
"""


# ---------------------------------------------------------------------------
# LaTeX rendering (same approach as publish_pdf.py)
# ---------------------------------------------------------------------------

_math_cache: dict[str, Path] = {}

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
    "font.family": "serif",
})

BLOCK_FONTSIZE = 11
INLINE_FONTSIZE = 11
RENDER_DPI = 150


def _render_latex(latex: str, fontsize: int, dpi: int, tmp_dir: Path,
                  is_block: bool = False) -> Path:
    """Render a LaTeX string to a transparent PNG."""
    key = f"{latex}|{fontsize}|{dpi}|{is_block}"
    if key in _math_cache:
        return _math_cache[key]

    h = hashlib.md5(key.encode()).hexdigest()[:12]
    out_path = tmp_dir / f"math_{h}.png"

    if is_block and r"\\" in latex:
        rendered = rf"$\begin{{gathered}}{latex}\end{{gathered}}$"
    else:
        rendered = rf"${latex}$"

    fig, ax = plt.subplots(figsize=(0.1, 0.1))
    ax.axis("off")
    ax.text(
        0.5, 0.5, rendered,
        fontsize=fontsize, ha="center", va="center",
        transform=ax.transAxes, color="#1a1a1a",
    )
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight",
                transparent=True, pad_inches=0.04)
    plt.close(fig)

    _math_cache[key] = out_path
    return out_path


def render_math_in_markdown(md_text: str, tmp_dir: Path) -> str:
    """Replace $$...$$ and $...$ with rendered PNG images."""

    # 0) Protect code blocks from math processing
    _code_blocks = []
    def _save_code(m):
        _code_blocks.append(m.group(0))
        return f"__CODE_BLOCK_{len(_code_blocks)-1}__"
    md_text = re.sub(r"```.*?```", _save_code, md_text, flags=re.DOTALL)
    md_text = re.sub(r"`[^`]+`", _save_code, md_text)

    # 1) Block math (multi-line between $$ fences)
    def _replace_block(m):
        latex = m.group(1).strip()
        png = _render_latex(latex, BLOCK_FONTSIZE, RENDER_DPI, tmp_dir,
                            is_block=True)
        return f'\n<div class="math-block"><img src="file://{png}"></div>\n'

    md_text = re.sub(r"\n\$\$\n(.*?)\n\$\$\n", _replace_block, md_text,
                     flags=re.DOTALL)

    # 2) Single-line display math
    def _replace_display(m):
        latex = m.group(1).strip()
        png = _render_latex(latex, BLOCK_FONTSIZE, RENDER_DPI, tmp_dir,
                            is_block=True)
        return f'\n<div class="math-block"><img src="file://{png}"></div>\n'

    md_text = re.sub(r"\n\$\$(.+?)\$\$\n", _replace_display, md_text)

    # 3) Inline math ($$...$$)
    def _replace_inline(m):
        latex = m.group(1).strip()
        png = _render_latex(latex, INLINE_FONTSIZE, RENDER_DPI, tmp_dir)
        return f'<span class="math-inline"><img src="file://{png}"></span>'

    md_text = re.sub(r"\$\$(.+?)\$\$", _replace_inline, md_text)

    # 4) Single-dollar inline math ($...$)
    # Skip currency patterns like $1.2B, $180M, $5.50, $360M/year
    _currency_re = re.compile(
        r"^\d[\d,]*\.?\d*\s*[KkMmBbTt%]|"   # $180M, $1.2B, $5.50
        r"^\d[\d,]*\.?\d*/",                  # $0.80/hr
        re.IGNORECASE,
    )

    def _replace_single(m):
        latex = m.group(1).strip()
        if not latex:
            return m.group(0)
        # Skip currency amounts
        if _currency_re.match(latex):
            return m.group(0)
        png = _render_latex(latex, INLINE_FONTSIZE, RENDER_DPI, tmp_dir)
        return f'<span class="math-inline"><img src="file://{png}"></span>'

    md_text = re.sub(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)",
                     _replace_single, md_text)

    # Restore code blocks
    for i, block in enumerate(_code_blocks):
        md_text = md_text.replace(f"__CODE_BLOCK_{i}__", block)

    return md_text


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def extract_title(text: str) -> str:
    """Extract H1 title from markdown."""
    match = re.match(r"^#\s+(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else "Case Study"


def md_to_pdf(case_study_dir: Path) -> Path:
    """Convert case_study.md to a styled PDF."""
    md_path = case_study_dir / "case_study.md"
    if not md_path.exists():
        print(f"Error: {md_path} not found")
        sys.exit(1)

    md_text = md_path.read_text(encoding="utf-8")
    title = extract_title(md_text)

    # Render LaTeX to images
    tmp_dir = Path(tempfile.mkdtemp(prefix="vizuara_cs_math_"))
    print("Rendering LaTeX equations...")
    md_text = render_math_in_markdown(md_text, tmp_dir)

    # Convert markdown to HTML
    html_body = markdown.markdown(
        md_text,
        extensions=["fenced_code", "codehilite", "tables", "toc"],
        extension_configs={
            "codehilite": {"guess_lang": True, "css_class": "highlight"}
        },
    )

    full_html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><style>{CSS}</style></head>
<body>{html_body}</body>
</html>"""

    pdf_path = case_study_dir / "case_study.pdf"

    print("Generating PDF...")
    HTML(string=full_html, base_url=str(case_study_dir)).write_pdf(str(pdf_path))

    # Clean up temp math images
    for f in tmp_dir.glob("*.png"):
        f.unlink()
    tmp_dir.rmdir()

    return pdf_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert a case study markdown to a styled PDF"
    )
    parser.add_argument(
        "case_study_dir",
        help="Path to case study directory (containing case_study.md)"
    )
    args = parser.parse_args()

    case_study_dir = Path(args.case_study_dir)
    if not case_study_dir.is_absolute():
        case_study_dir = PROJECT_ROOT / case_study_dir

    pdf_path = md_to_pdf(case_study_dir)
    print(f"PDF generated: {pdf_path}")


if __name__ == "__main__":
    main()