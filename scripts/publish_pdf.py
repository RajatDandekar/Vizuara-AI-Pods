#!/usr/bin/env python3
"""
Publish a markdown article as a styled PDF.

Usage:
    python scripts/publish_pdf.py <article_dir> [--name "Custom Name"]

Examples:
    python scripts/publish_pdf.py output/articles/world-action-models
    python scripts/publish_pdf.py output/articles/world-action-models --name "World Action Models"

The script will:
  1. Read final.md from the article directory
  2. Render LaTeX math as images via real LaTeX engine
  3. Resolve local figure paths (figures/*.png)
  4. Render a styled PDF named "<ConceptName>_Claude.pdf"
  5. Save it to output/pdf/
"""

import argparse
import hashlib
import re
import sys
import tempfile
from pathlib import Path

# Ensure we can import from the venv
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
OUTPUT_DIR = PROJECT_ROOT / "output" / "pdf"

CSS = """
@page {
    size: A4;
    margin: 1.8cm 1.8cm 1.8cm 1.8cm;
    @bottom-center {
        content: "Vizuara AI";
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
    font-size: 11.5pt;
    line-height: 1.5;
    color: #1a1a1a;
    max-width: 100%;
}

h1 {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 26pt;
    font-weight: 700;
    color: #111;
    margin-top: 0;
    margin-bottom: 0.2em;
    line-height: 1.2;
}

h1 + p:first-of-type {
    font-size: 13pt;
    color: #555;
    font-style: italic;
    margin-bottom: 1em;
    border-bottom: 2px solid #e0e0e0;
    padding-bottom: 0.6em;
}

h2 {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 18pt;
    font-weight: 700;
    color: #222;
    margin-top: 1.2em;
    margin-bottom: 0.4em;
    border-bottom: 1px solid #e8e8e8;
    padding-bottom: 0.2em;
}

h3 {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 14pt;
    font-weight: 600;
    color: #333;
    margin-top: 1.0em;
    margin-bottom: 0.3em;
}

p {
    margin-bottom: 0.5em;
    text-align: left;
}

strong { font-weight: 700; }
em { font-style: italic; }

/* Figure + caption wrapper */
.figure-group {
    margin: 0.4em 0 0.2em 0;
}

/* Article figures — constrained width, tight to caption */
.figure-block {
    text-align: center;
    margin: 0 0 0.1em 0;
}
.figure-block img {
    max-width: 55%;
    height: auto;
    display: inline-block;
    border-radius: 4px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

/* Figure captions — sits right under image, minimal gap below */
.fig-caption {
    text-align: center;
    font-size: 9.5pt;
    font-style: italic;
    color: #666;
    margin: 0.1em 0 0.2em 0;
    padding: 0 2em;
}

/* No break-before constraints — let figures flow naturally after text */

/* References section — each reference on its own line */
.references-section {
    margin-top: 1.5em;
    padding-top: 0.8em;
    border-top: 1px solid #e0e0e0;
}
.references-section p,
.references-section li {
    font-size: 9.5pt;
    color: #444;
    margin-bottom: 0.25em;
    line-height: 1.4;
    text-indent: -1.5em;
    padding-left: 1.5em;
}
.references-section ul,
.references-section ol {
    list-style: none;
    padding-left: 0;
}
.references-section li {
    list-style: none;
}

/* Block math (display equations) */
.math-block {
    text-align: center;
    margin: 0.4em 0;
}
.math-block img {
    display: inline-block;
    height: auto;
    vertical-align: middle;
}

/* Equation images (referenced as equation PNGs, not rendered via LaTeX engine) */
.equation-block {
    text-align: center;
    margin: 0.3em 0;
    break-inside: avoid;
}
.equation-block img {
    max-width: 65%;
    max-height: 3.5em;
    height: auto;
    display: inline-block;
    vertical-align: middle;
}

/* Inline math — scale to match surrounding text */
.math-inline img {
    display: inline;
    height: 1.3em;
    vertical-align: -0.35em;
    margin: 0 0.05em;
}

pre {
    background: #f6f8fa;
    border: 1px solid #e1e4e8;
    border-radius: 6px;
    padding: 1em;
    font-size: 9.5pt;
    line-height: 1.5;
    overflow-x: auto;
    margin: 0.6em 0;
}

code {
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 9.5pt;
    background: #f0f2f4;
    padding: 0.15em 0.35em;
    border-radius: 3px;
}

pre code {
    background: none;
    padding: 0;
    border-radius: 0;
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
    margin: 2em 0;
}

ol, ul {
    margin-bottom: 0.5em;
    padding-left: 1.5em;
}

li { margin-bottom: 0.3em; }
"""


# ---------------------------------------------------------------------------
# LaTeX rendering via real LaTeX engine (matplotlib with usetex=True)
# ---------------------------------------------------------------------------

_math_cache: dict[str, Path] = {}

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
    "font.family": "serif",
})

# Font sizes tuned for A4 PDF
BLOCK_FONTSIZE = 11
INLINE_FONTSIZE = 11
RENDER_DPI = 150


def _render_latex(latex: str, fontsize: int, dpi: int, tmp_dir: Path, is_block: bool = False) -> Path:
    """Render a LaTeX string to a transparent PNG. Returns the file path."""
    key = f"{latex}|{fontsize}|{dpi}|{is_block}"
    if key in _math_cache:
        return _math_cache[key]

    h = hashlib.md5(key.encode()).hexdigest()[:12]
    out_path = tmp_dir / f"math_{h}.png"

    # For multi-line block equations, wrap in gathered environment
    if is_block and r"\\" in latex:
        rendered = rf"$\begin{{gathered}}{latex}\end{{gathered}}$"
    else:
        rendered = rf"${latex}$"

    fig, ax = plt.subplots(figsize=(0.1, 0.1))
    ax.axis("off")
    ax.text(
        0.5, 0.5,
        rendered,
        fontsize=fontsize,
        ha="center", va="center",
        transform=ax.transAxes,
        color="#1a1a1a",
    )
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight", transparent=True, pad_inches=0.04)
    plt.close(fig)

    _math_cache[key] = out_path
    return out_path


def render_math_in_markdown(md_text: str, tmp_dir: Path) -> str:
    """Replace $$...$$ patterns with <img> tags pointing to rendered PNGs.

    Handles three patterns:
      1. Block math:  \\n$$\\n ... \\n$$\\n
      2. Single-line display: $$...$$  as the only content on a line
      3. Inline math:  $$...$$  within running text
    """

    # 1) Block math (multi-line between $$ fences)
    def _replace_block(m):
        latex = m.group(1).strip()
        png = _render_latex(latex, fontsize=BLOCK_FONTSIZE, dpi=RENDER_DPI, tmp_dir=tmp_dir, is_block=True)
        return f'\n<div class="math-block"><img src="file://{png}"></div>\n'

    md_text = re.sub(
        r"\n\$\$\n(.*?)\n\$\$\n",
        _replace_block,
        md_text,
        flags=re.DOTALL,
    )

    # 2) Single-line display math (entire line is $$...$$)
    def _replace_single_line_display(m):
        latex = m.group(1).strip()
        png = _render_latex(latex, fontsize=BLOCK_FONTSIZE, dpi=RENDER_DPI, tmp_dir=tmp_dir, is_block=True)
        return f'\n<div class="math-block"><img src="file://{png}"></div>\n'

    md_text = re.sub(
        r"\n\$\$(.+?)\$\$\n",
        _replace_single_line_display,
        md_text,
    )

    # 3) Inline math ($$...$$ within text)
    def _replace_inline(m):
        latex = m.group(1).strip()
        png = _render_latex(latex, fontsize=INLINE_FONTSIZE, dpi=RENDER_DPI, tmp_dir=tmp_dir)
        return f'<span class="math-inline"><img src="file://{png}"></span>'

    md_text = re.sub(r"\$\$(.+?)\$\$", _replace_inline, md_text)

    # 4) Single-dollar inline math ($...$) — must run AFTER $$ patterns
    def _replace_single_inline(m):
        latex = m.group(1).strip()
        if not latex:
            return m.group(0)
        png = _render_latex(latex, fontsize=INLINE_FONTSIZE, dpi=RENDER_DPI, tmp_dir=tmp_dir)
        return f'<span class="math-inline"><img src="file://{png}"></span>'

    md_text = re.sub(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", _replace_single_inline, md_text)

    return md_text


# ---------------------------------------------------------------------------
# Figure handling
# ---------------------------------------------------------------------------

def resolve_figure_paths(html: str, article_dir: Path) -> str:
    """Convert relative figure paths to absolute file:// URIs, wrap in .figure-block divs."""
    figures_dir = article_dir / "figures"

    def replace_img(match):
        full_tag = match.group(0)
        src = match.group(1)

        # Skip math images (already absolute file:// paths)
        if src.startswith("file://"):
            return full_tag

        # Determine if this is an equation image or a figure image
        is_equation = src.startswith("equations/")

        # Resolve relative path
        if src.startswith("figures/"):
            abs_path = figures_dir / src.replace("figures/", "")
        else:
            abs_path = article_dir / src

        if abs_path.exists():
            img_class = "equation" if is_equation else "figure"
            return full_tag.replace(f'src="{src}"', f'src="file://{abs_path}" class="{img_class}"')
        return full_tag

    return re.sub(r'<img[^>]*src="([^"]+)"[^>]*/?\s*>', replace_img, html)


def style_figures_and_captions(html: str) -> str:
    """Wrap figure images + their captions into tight figure-block + fig-caption divs,
    all inside a .figure-group to keep them together across page breaks."""
    # Pattern 1: img + caption in the SAME <p> (markdown puts them together)
    html = re.sub(
        r'<p>\s*(<img[^>]*class="figure"[^>]*/?\s*>)\s*\n?\s*<em>(.*?)</em>\s*</p>',
        r'<div class="figure-group"><div class="figure-block">\1</div>\n<div class="fig-caption">\2</div></div>',
        html,
        flags=re.DOTALL,
    )
    # Pattern 2: img and caption in separate <p> tags
    html = re.sub(
        r'<p>\s*(<img[^>]*class="figure"[^>]*/?\s*>)\s*</p>\s*<p>\s*<em>(.*?)</em>\s*</p>',
        r'<div class="figure-group"><div class="figure-block">\1</div>\n<div class="fig-caption">\2</div></div>',
        html,
        flags=re.DOTALL,
    )
    # Pattern 3: figure without caption
    html = re.sub(
        r'<p>\s*(<img[^>]*class="figure"[^>]*/?\s*>)\s*</p>',
        r'<div class="figure-group"><div class="figure-block">\1</div></div>',
        html,
    )
    return html


def style_equation_images(html: str) -> str:
    """Wrap equation PNG images in .equation-block divs for compact styling."""
    # Match <p> tags containing only an equation image (class="equation")
    html = re.sub(
        r'<p>\s*(<img[^>]*class="equation"[^>]*/?\s*>)\s*</p>',
        r'<div class="equation-block">\1</div>',
        html,
    )
    return html


def style_references_section(html: str) -> str:
    """Detect the references section and wrap it with .references-section for line-by-line styling."""
    # Case 1: Heading-based references (h2 or h3 with "References")
    for tag in ['h2', 'h3']:
        pattern = rf'(<{tag}[^>]*>.*?[Rr]eferences.*?</{tag}>)'
        match = re.search(pattern, html, flags=re.DOTALL)
        if match:
            start = match.start()
            after = match.end()
            # Find the next h2/h3 or </body> to close the section
            rest = html[after:]
            close = re.search(r'<h[23]', rest)
            if close:
                end = after + close.start()
            else:
                body_end = html.find('</body>', after)
                end = body_end if body_end != -1 else len(html)
            html = (
                html[:start]
                + '<div class="references-section">'
                + html[start:end]
                + '</div>'
                + html[end:]
            )
            return html

    # Case 2: Italic/bold "References:" without a heading (e.g. *References:*)
    ref_match = re.search(
        r'(<p>\s*(?:<em>|<strong>)\s*References\s*:?\s*(?:</em>|</strong>)\s*</p>)',
        html, flags=re.IGNORECASE
    )
    if ref_match:
        start = ref_match.start()
        rest = html[start:]
        # Wrap from "References:" to the next heading or end
        close = re.search(r'<h[23]', rest[ref_match.end() - start:])
        if close:
            end = start + (ref_match.end() - start) + close.start()
        else:
            body_end = html.find('</body>', start)
            end = body_end if body_end != -1 else len(html)
        html = (
            html[:start]
            + '<div class="references-section">'
            + html[start:end]
            + '</div>'
            + html[end:]
        )

    return html


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def extract_title_from_md(text: str) -> str:
    """Extract the H1 title from markdown."""
    match = re.match(r"^#\s+(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else "Article"


def md_to_pdf(article_dir: Path, concept_name: str | None = None) -> Path:
    """Convert a markdown article to a styled PDF."""
    md_path = article_dir / "final.md"
    if not md_path.exists():
        print(f"Error: {md_path} not found")
        sys.exit(1)

    md_text = md_path.read_text(encoding="utf-8")

    # Strip the "Vizuara AI" byline
    md_text = re.sub(r"\nVizuara AI\n", "\n", md_text)

    # Derive concept name from title if not provided
    if not concept_name:
        concept_name = extract_title_from_md(md_text)

    # Render LaTeX to images
    tmp_dir = Path(tempfile.mkdtemp(prefix="vizuara_math_"))
    print("Rendering LaTeX equations...")
    md_text = render_math_in_markdown(md_text, tmp_dir)

    # Convert markdown to HTML
    html_body = markdown.markdown(
        md_text,
        extensions=["fenced_code", "codehilite", "tables", "toc"],
        extension_configs={"codehilite": {"guess_lang": True, "css_class": "highlight"}},
    )

    # Resolve figure paths and add figure class
    html_body = resolve_figure_paths(html_body, article_dir)

    # Style equation images (compact, no caption) — must run before figure styling
    html_body = style_equation_images(html_body)

    # Wrap figures and captions into tight blocks
    html_body = style_figures_and_captions(html_body)

    # Style references section for line-by-line display
    html_body = style_references_section(html_body)

    full_html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><style>{CSS}</style></head>
<body>{html_body}</body>
</html>"""

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate filename
    safe_name = concept_name.replace(":", " -").strip()
    pdf_name = f"{safe_name}_Claude.pdf"
    pdf_path = OUTPUT_DIR / pdf_name

    # Render PDF
    print("Generating PDF...")
    HTML(string=full_html, base_url=str(article_dir)).write_pdf(str(pdf_path))

    # Clean up temp math images
    for f in tmp_dir.glob("*.png"):
        f.unlink()
    tmp_dir.rmdir()

    return pdf_path


def main():
    parser = argparse.ArgumentParser(description="Publish a markdown article as a styled PDF")
    parser.add_argument("article_dir", help="Path to article directory (containing final.md)")
    parser.add_argument("--name", help="Concept name for the PDF filename (default: extracted from title)")
    args = parser.parse_args()

    article_dir = Path(args.article_dir)
    if not article_dir.is_absolute():
        article_dir = PROJECT_ROOT / article_dir

    pdf_path = md_to_pdf(article_dir, args.name)
    print(f"PDF generated: {pdf_path}")


if __name__ == "__main__":
    main()