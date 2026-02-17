#!/usr/bin/env python3
"""
Figure generation for Vizuara Substack articles.

Supports five methods:
1. PaperBanana — Multi-agent pipeline for publication-quality academic figures
2. Gemini — Google's image generation API directly
3. GPT — OpenAI's gpt-image-1 model
4. Matplotlib — Fallback for data plots and simple diagrams
5. Shootout — Run all backends, save side-by-side for comparison

Usage:
    # Auto mode (uses figure_preferences.json routing, falls back to gemini)
    python generate_figure.py --description "..." --output figure.png

    # Specific backend
    python generate_figure.py --description "..." --output figure.png --method gpt

    # Run all backends for comparison
    python generate_figure.py --description "..." --output figure.png --method shootout

Requirements:
    pip install python-dotenv matplotlib Pillow google-generativeai openai

Optional:
    pip install paperbanana   (for PaperBanana pipeline)

Environment:
    GOOGLE_API_KEY in your project's .env.local
    OPENAI_API_KEY in your project's .env.local (for GPT method)
"""

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_loader import load_env, get_google_api_key

# Path to figure preferences (built up by shootout comparisons)
PREFS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure_preferences.json")


def load_preferences() -> dict:
    """Load figure-type → preferred-method routing map."""
    if os.path.exists(PREFS_PATH):
        with open(PREFS_PATH) as f:
            return json.load(f)
    return {}


def generate_with_paperbanana(description: str, output_path: str, style: str):
    """
    Use PaperBanana's multi-agent pipeline for academic figures.

    Pipeline: Retrieve → Plan → Style → Render → Critique (2-3 rounds)
    Best for: Methodology diagrams, architecture figures, academic illustrations.
    """
    try:
        from paperbanana import PaperBananaPipeline, GenerationInput, DiagramType
        from paperbanana.core.config import Settings

        settings = Settings(
            vlm_model="gemini-2.0-flash",
            image_model="gemini-3-pro-image-preview",
            refinement_iterations=2,
            output_dir=os.path.dirname(output_path) or ".",
        )

        pipeline = PaperBananaPipeline(settings=settings)

        async def run():
            return await pipeline.generate(
                GenerationInput(
                    source_context=description,
                    communicative_intent=style or description[:200],
                    diagram_type=DiagramType.METHODOLOGY,
                )
            )

        result = asyncio.run(run())

        if result.image_path and os.path.exists(result.image_path):
            import shutil
            shutil.copy2(result.image_path, output_path)
            print(f"Figure generated via PaperBanana: {output_path}")
            print(f"  Refinement iterations: {len(result.iterations)}")
            return output_path

        print("PaperBanana returned no image. Trying next method...")
        return None

    except ImportError:
        print("PaperBanana not installed. Run: pip install paperbanana")
        print("Trying next method...")
        return None
    except Exception as e:
        print(f"PaperBanana failed: {e}. Trying next method...")
        return None


def generate_with_gemini(description: str, output_path: str):
    """
    Use Gemini's image generation API directly.

    Good for: Architecture diagrams, concept illustrations, flowcharts.
    """
    try:
        import google.generativeai as genai

        api_key = get_google_api_key()
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel("gemini-2.5-flash-image")

        prompt = f"""Generate a clean, publication-ready technical diagram:

{description}

Style requirements:
- Clean white background
- Professional color palette (blues, teals, grays)
- Clear, readable labels with good font sizes
- No visual clutter or unnecessary decorations
- Suitable for a technical blog post (Substack)
- High contrast for readability on screens"""

        response = model.generate_content(prompt)

        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    with open(output_path, "wb") as f:
                        f.write(part.inline_data.data)
                    print(f"Figure generated via Gemini: {output_path}")
                    return output_path

        print("Gemini returned no image data. Trying next method...")
        return None

    except ImportError:
        print("google-generativeai not installed. Run: pip install google-generativeai")
        return None
    except Exception as e:
        print(f"Gemini generation failed: {e}. Trying next method...")
        return None


def generate_with_gpt(description: str, output_path: str):
    """
    Use OpenAI's gpt-image-1 model.

    Good for: Photorealistic images, detailed concept art, complex scenes.
    """
    try:
        from openai import OpenAI
        import base64

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("OPENAI_API_KEY not set. Trying next method...")
            return None

        client = OpenAI(api_key=api_key)

        prompt = f"""Generate a clean, publication-ready technical diagram for a blog article:

{description}

Style: White background, professional blues and teals, clear readable labels,
no clutter, high contrast, suitable for a technical blog post."""

        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            n=1,
            size="1024x1024",
        )

        if response.data and response.data[0].b64_json:
            img_data = base64.b64decode(response.data[0].b64_json)
            with open(output_path, "wb") as f:
                f.write(img_data)
            print(f"Figure generated via GPT: {output_path}")
            return output_path
        elif response.data and response.data[0].url:
            import urllib.request
            urllib.request.urlretrieve(response.data[0].url, output_path)
            print(f"Figure generated via GPT: {output_path}")
            return output_path

        print("GPT returned no image data. Trying next method...")
        return None

    except ImportError:
        print("openai not installed. Run: pip install openai")
        return None
    except Exception as e:
        print(f"GPT generation failed: {e}. Trying next method...")
        return None


def generate_with_matplotlib(description: str, output_path: str):
    """
    Fallback using Matplotlib.

    Best for: Statistical plots, bar charts, line graphs, heatmaps.
    For complex diagrams, this produces a placeholder.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis("off")

        box = mpatches.FancyBboxPatch(
            (0.5, 0.5), 9, 5,
            boxstyle="round,pad=0.3",
            facecolor="#f0f4f8",
            edgecolor="#4a90d9",
            linewidth=2,
        )
        ax.add_patch(box)

        ax.text(5, 4.0, "FIGURE PLACEHOLDER", ha="center", va="center",
                fontsize=14, fontweight="bold", color="#4a90d9")
        ax.text(5, 2.5, description, ha="center", va="center",
                fontsize=9, color="#333333", wrap=True, style="italic")
        ax.text(5, 0.8, "Replace with specific matplotlib code or regenerate with another backend",
                ha="center", va="center", fontsize=8, color="#888888")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close()

        print(f"Matplotlib placeholder generated: {output_path}")
        return output_path

    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib", file=sys.stderr)
        return None


def run_shootout(description: str, output_path: str, style: str):
    """Run all backends and save results side-by-side."""
    shootout_dir = os.path.splitext(output_path)[0] + "_shootout"
    os.makedirs(shootout_dir, exist_ok=True)

    from figure_shootout import run_shootout as _shootout
    results = _shootout(description, shootout_dir, caption=style)

    # If any succeeded, copy the first successful one to the main output path
    for method in ["paperbanana", "gpt", "gemini", "matplotlib"]:
        candidate = os.path.join(shootout_dir, f"{method}.png")
        if results.get(method, {}).get("status") == "success" and os.path.exists(candidate):
            import shutil
            shutil.copy2(candidate, output_path)
            print(f"\nDefault output (from {method}): {output_path}")
            print(f"Compare all results in: {shootout_dir}/")
            return output_path

    print("All methods failed in shootout.")
    return None


# Method dispatch table
METHODS = {
    "paperbanana": lambda desc, out, style: generate_with_paperbanana(desc, out, style),
    "gemini": lambda desc, out, style: generate_with_gemini(desc, out),
    "gpt": lambda desc, out, style: generate_with_gpt(desc, out),
    "matplotlib": lambda desc, out, style: generate_with_matplotlib(desc, out),
    "shootout": run_shootout,
}

# Auto fallback order
AUTO_ORDER = ["paperbanana", "gemini", "matplotlib"]


def main():
    parser = argparse.ArgumentParser(
        description="Generate figures for Vizuara Substack articles"
    )
    parser.add_argument(
        "--description", required=True,
        help="Detailed description of the figure to generate"
    )
    parser.add_argument(
        "--style", default="clean, publication-ready, Vizuara aesthetic",
        help="Style guidance for the figure"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output file path (e.g., output/articles/topic/figures/figure_1.png)"
    )
    parser.add_argument(
        "--method",
        choices=["paperbanana", "gemini", "gpt", "matplotlib", "shootout", "auto"],
        default="auto",
        help="Generation method. 'auto' uses preferences, 'shootout' runs all backends"
    )

    args = parser.parse_args()

    load_env()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    result = None

    if args.method == "shootout":
        result = run_shootout(args.description, args.output, args.style)
    elif args.method == "auto":
        for method_name in AUTO_ORDER:
            print(f"Attempting {method_name}...")
            fn = METHODS[method_name]
            result = fn(args.description, args.output, args.style)
            if result:
                break
    else:
        fn = METHODS[args.method]
        result = fn(args.description, args.output, args.style)

    if result:
        print(f"\nFigure saved to: {args.output}")
    else:
        print("\nERROR: All figure generation methods failed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()