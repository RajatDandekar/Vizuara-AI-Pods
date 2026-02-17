#!/usr/bin/env python3
"""
Assemble final Substack-ready markdown from draft + generated assets.

Replaces {{FIGURE: ...}} and {{EQUATION: ...}} placeholders in the draft
with actual image references or inline LaTeX.

Requirements:
    pip install python-dotenv
"""

import argparse
import os
import re
import glob
import sys

# Add the scripts directory to path so we can import env_loader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env_loader import load_env


def assemble(
    draft_path: str,
    figures_dir: str,
    equations_dir: str,
    output_path: str,
    use_inline_latex: bool = True,
):
    """
    Assemble the final article by replacing placeholders with actual content.
    
    Args:
        draft_path: Path to the draft markdown with placeholders
        figures_dir: Directory containing generated figure PNGs
        equations_dir: Directory containing rendered equation PNGs
        output_path: Where to write the final assembled markdown
        use_inline_latex: If True, keep LaTeX as $$...$$ for Substack's 
                         native rendering. If False, use rendered PNG images.
    """
    if not os.path.exists(draft_path):
        print(f"ERROR: Draft not found at {draft_path}", file=sys.stderr)
        sys.exit(1)

    with open(draft_path, "r") as f:
        content = f.read()

    # --- Replace figure placeholders ---
    # Build a lookup from figure number -> file path for explicit matching
    figure_files = sorted(glob.glob(os.path.join(figures_dir, "figure_*.png")))
    figure_lookup = {}
    for fp in figure_files:
        fname = os.path.basename(fp)
        m = re.match(r"figure_(\d+)", fname)
        if m:
            figure_lookup[int(m.group(1))] = fp

    figure_idx = [0]  # mutable counter for sequential assignment

    def replace_figure(match):
        figure_idx[0] += 1
        n = figure_idx[0]  # 1-based figure number
        raw = match.group(1).strip()

        # Support optional short caption via || separator:
        #   {{FIGURE: long description for generator || Short caption for readers}}
        if "||" in raw:
            description, caption = raw.split("||", 1)
            description = description.strip()
            caption = caption.strip()
        else:
            description = raw
            caption = None  # will use generic "Figure N" below

        # Try explicit numbered match first, then fall back to sequential
        fig_path = figure_lookup.get(n)
        if fig_path is None:
            # Fallback: try 0-based numbering
            fig_path = figure_lookup.get(n - 1)

        if fig_path is not None:
            try:
                rel_path = os.path.relpath(fig_path, os.path.dirname(output_path))
            except ValueError:
                rel_path = fig_path

            # Alt text: short caption or generic label (never the full description)
            alt_text = caption if caption else f"Figure {n}"
            # Caption below image: short caption or generic label
            display_caption = caption if caption else f"Figure {n}"
            return f"\n![{alt_text}]({rel_path})\n*{display_caption}*\n"
        else:
            return f"\n**[FIGURE MISSING #{n}: {description}]**\n"

    content = re.sub(
        r"\{\{FIGURE:\s*(.*?)\}\}", replace_figure, content, flags=re.DOTALL
    )

    # --- Replace equation placeholders ---
    if use_inline_latex:
        # Keep LaTeX inline — Substack renders $$...$$ natively via KaTeX
        def replace_equation_inline(match):
            latex = match.group(1).strip()
            return f"\n$$\n{latex}\n$$\n"

        content = re.sub(
            r"\{\{EQUATION:\s*(.*?)\}\}", replace_equation_inline, content, flags=re.DOTALL
        )
    else:
        # Use rendered PNG images
        eq_files = sorted(glob.glob(os.path.join(equations_dir, "eq_*.png")))
        eq_idx = 0

        def replace_equation_image(match):
            nonlocal eq_idx
            latex = match.group(1).strip()

            if eq_idx < len(eq_files):
                eq_path = eq_files[eq_idx]
                eq_idx += 1

                try:
                    rel_path = os.path.relpath(eq_path, os.path.dirname(output_path))
                except ValueError:
                    rel_path = eq_path

                return f"\n![Equation]({rel_path})\n"
            else:
                # Fallback to inline LaTeX if image is missing
                return f"\n$$\n{latex}\n$$\n"

        content = re.sub(
            r"\{\{EQUATION:\s*(.*?)\}\}", replace_equation_image, content, flags=re.DOTALL
        )

    # --- Write final article ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write(content)

    # --- Report stats ---
    word_count = len(content.split())
    num_figures = content.count("![")
    num_code_blocks = content.count("```") // 2
    num_inline_equations = content.count("$$") // 2
    num_missing = content.count("[FIGURE MISSING")

    print(f"\nArticle assembled: {output_path}")
    print(f"  Words:           ~{word_count}")
    print(f"  Figures:         {num_figures}")
    print(f"  Code blocks:     {num_code_blocks}")
    print(f"  Equations:       {num_inline_equations}")

    if num_missing > 0:
        print(f"  ⚠ Missing figs:  {num_missing}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Assemble Vizuara Substack article from draft and assets"
    )
    parser.add_argument(
        "--draft", required=True,
        help="Path to draft markdown with {{FIGURE:}} and {{EQUATION:}} placeholders"
    )
    parser.add_argument(
        "--figures-dir", required=True,
        help="Directory containing generated figure_N.png files"
    )
    parser.add_argument(
        "--equations-dir", required=True,
        help="Directory containing rendered eq_N.png files"
    )
    parser.add_argument(
        "--output", required=True,
        help="Path for the final assembled markdown"
    )
    parser.add_argument(
        "--render-equations", action="store_true", default=False,
        help="Use rendered PNG images for equations instead of inline LaTeX"
    )

    args = parser.parse_args()

    # Load environment
    load_env()

    assemble(
        draft_path=args.draft,
        figures_dir=args.figures_dir,
        equations_dir=args.equations_dir,
        output_path=args.output,
        use_inline_latex=not args.render_equations,
    )


if __name__ == "__main__":
    main()