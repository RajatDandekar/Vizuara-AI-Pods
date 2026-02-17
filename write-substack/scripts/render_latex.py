#!/usr/bin/env python3
"""
Render LaTeX equations to PNG images for Vizuara Substack articles.

Methods (tried in order with 'auto'):
1. Full LaTeX — Highest quality, requires texlive installation
2. Matplotlib — No dependencies beyond matplotlib, good enough for most equations

Requirements:
    pip install python-dotenv matplotlib Pillow

Optional (for higher quality):
    brew install --cask mactex-no-gui   (macOS)
    brew install poppler                 (macOS, for pdf→png conversion)
"""

import argparse
import os
import sys

# Add the scripts directory to path so we can import env_loader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env_loader import load_env


def render_with_matplotlib(latex: str, output_path: str, fontsize: int = 20):
    """
    Render LaTeX using matplotlib's built-in mathtext engine.
    
    Works without any LaTeX installation. Handles most common math notation.
    Limitations: Some advanced LaTeX packages (tikz, etc.) won't work.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(0.1, 0.1))
    ax.axis("off")

    # Render the equation
    text = ax.text(
        0.5, 0.5, f"${latex}$",
        fontsize=fontsize,
        ha="center", va="center",
        transform=ax.transAxes,
        color="#1a1a1a",
        math_fontfamily="cm",  # Computer Modern — standard LaTeX font
    )

    # Save with tight bounding box
    fig.savefig(
        output_path, dpi=200,
        bbox_inches="tight",
        pad_inches=0.15,
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )
    plt.close()

    # Trim excess whitespace for clean embedding
    try:
        from PIL import Image, ImageOps

        img = Image.open(output_path)
        img = img.convert("RGB")

        # Auto-crop white borders
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)

        # Add small uniform padding
        img = ImageOps.expand(img, border=15, fill="white")
        img.save(output_path, dpi=(200, 200))

    except ImportError:
        pass  # Pillow not available, skip trimming

    return output_path


def render_with_latex(latex: str, output_path: str):
    """
    Render using full LaTeX installation for highest quality.
    
    Supports all LaTeX packages: amsmath, amssymb, tikz, etc.
    Requires: texlive (or mactex on macOS) and poppler (pdftoppm).
    """
    import subprocess
    import tempfile
    import shutil

    # Check if pdflatex is available
    if not shutil.which("pdflatex"):
        print("pdflatex not found. Install texlive or mactex.")
        return None

    if not shutil.which("pdftoppm"):
        print("pdftoppm not found. Install poppler: brew install poppler")
        return None

    tex_content = r"""\documentclass[preview,border=3pt]{standalone}
\usepackage{amsmath,amssymb,amsfonts,mathtools}
\usepackage{xcolor}
\definecolor{textcolor}{HTML}{1a1a1a}
\begin{document}
\color{textcolor}
\large
$\displaystyle """ + latex + r"""$
\end{document}
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tex_file = os.path.join(tmpdir, "equation.tex")
        with open(tex_file, "w") as f:
            f.write(tex_content)

        # Compile LaTeX → PDF
        result = subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-output-directory", tmpdir,
                tex_file,
            ],
            capture_output=True, text=True,
        )

        if result.returncode != 0:
            print(f"LaTeX compilation failed:\n{result.stderr}", file=sys.stderr)
            return None

        pdf_file = os.path.join(tmpdir, "equation.pdf")

        if not os.path.exists(pdf_file):
            print("PDF not generated.", file=sys.stderr)
            return None

        # Convert PDF → PNG
        png_prefix = os.path.join(tmpdir, "equation")
        subprocess.run(
            ["pdftoppm", "-png", "-r", "300", "-singlefile", pdf_file, png_prefix],
            capture_output=True,
        )

        png_file = png_prefix + ".png"
        if os.path.exists(png_file):
            shutil.move(png_file, output_path)

            # Trim whitespace with Pillow if available
            try:
                from PIL import Image, ImageOps

                img = Image.open(output_path).convert("RGB")
                bbox = img.getbbox()
                if bbox:
                    img = img.crop(bbox)
                img = ImageOps.expand(img, border=15, fill="white")
                img.save(output_path, dpi=(300, 300))
            except ImportError:
                pass

            return output_path

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Render LaTeX equations to PNG for Substack articles"
    )
    parser.add_argument(
        "--latex", required=True,
        help="LaTeX equation string (without $ delimiters)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output PNG file path"
    )
    parser.add_argument(
        "--method", choices=["matplotlib", "latex", "auto"],
        default="auto",
        help="Rendering method. 'auto' tries latex first, then matplotlib"
    )
    parser.add_argument(
        "--fontsize", type=int, default=20,
        help="Font size for matplotlib renderer (default: 20)"
    )

    args = parser.parse_args()

    # Load environment (not strictly needed for LaTeX, but keeps pattern consistent)
    load_env()

    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    result = None

    if args.method == "auto":
        print("Attempting full LaTeX render...")
        result = render_with_latex(args.latex, args.output)

        if not result:
            print("Falling back to matplotlib renderer...")
            result = render_with_matplotlib(args.latex, args.output, args.fontsize)

    elif args.method == "latex":
        result = render_with_latex(args.latex, args.output)
    elif args.method == "matplotlib":
        result = render_with_matplotlib(args.latex, args.output, args.fontsize)

    if result:
        print(f"Equation rendered: {args.output}")
    else:
        print("ERROR: Equation rendering failed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()