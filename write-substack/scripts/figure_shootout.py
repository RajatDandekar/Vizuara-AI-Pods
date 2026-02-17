#!/usr/bin/env python3
"""
Figure Shootout — Generate the same figure with ALL available backends.

Saves results side-by-side for visual comparison. After comparing,
record your preference in figure_preferences.json.

Usage:
    python figure_shootout.py \
      --description "A flowchart showing X -> Y -> Z" \
      --output-dir output/articles/topic/figures/shootout/figure_1 \
      [--caption "Optional caption for PaperBanana's communicative_intent"]

Each backend writes to:
    output_dir/
        gemini.png
        gpt.png
        paperbanana.png
        matplotlib.png
        _results.json    (metadata: timing, errors, etc.)
"""

import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_loader import load_env, get_google_api_key


def generate_gemini(description: str, output_path: str) -> dict:
    """Generate figure using Gemini's image generation."""
    t0 = time.time()
    try:
        import google.generativeai as genai

        genai.configure(api_key=get_google_api_key())
        model = genai.GenerativeModel("gemini-2.5-flash-image")

        prompt = f"""Generate a clean, publication-ready technical diagram:

{description}

Style requirements:
- Clean white background
- Professional color palette (blues, teals, grays)
- Clear, readable labels with good font sizes
- No visual clutter or unnecessary decorations
- Suitable for a technical blog post
- High contrast for readability on screens"""

        response = model.generate_content(prompt)
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    with open(output_path, "wb") as f:
                        f.write(part.inline_data.data)
                    return {"status": "success", "time_s": round(time.time() - t0, 1)}

        return {"status": "error", "error": "No image data returned", "time_s": round(time.time() - t0, 1)}
    except Exception as e:
        return {"status": "error", "error": str(e), "time_s": round(time.time() - t0, 1)}


def generate_gpt(description: str, output_path: str) -> dict:
    """Generate figure using OpenAI's gpt-image-1."""
    t0 = time.time()
    try:
        from openai import OpenAI
        import base64

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return {"status": "skipped", "error": "OPENAI_API_KEY not set", "time_s": 0}

        client = OpenAI(api_key=api_key)

        prompt = f"""Generate a clean, publication-ready technical diagram for a blog article:

{description}

Style: White background, professional blues and teals, clear readable labels, no clutter, high contrast."""

        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            n=1,
            size="1024x1024",
        )

        # gpt-image-1 returns base64
        if response.data and response.data[0].b64_json:
            img_data = base64.b64decode(response.data[0].b64_json)
            with open(output_path, "wb") as f:
                f.write(img_data)
            return {"status": "success", "time_s": round(time.time() - t0, 1)}
        elif response.data and response.data[0].url:
            import urllib.request
            urllib.request.urlretrieve(response.data[0].url, output_path)
            return {"status": "success", "time_s": round(time.time() - t0, 1)}

        return {"status": "error", "error": "No image data returned", "time_s": round(time.time() - t0, 1)}
    except Exception as e:
        return {"status": "error", "error": str(e), "time_s": round(time.time() - t0, 1)}


def generate_paperbanana(description: str, caption: str, output_path: str) -> dict:
    """Generate figure using PaperBanana's multi-agent pipeline."""
    t0 = time.time()
    try:
        from paperbanana import PaperBananaPipeline, GenerationInput, DiagramType
        from paperbanana.core.config import Settings

        settings = Settings(
            vlm_model="gemini-2.0-flash",
            image_model="gemini-3-pro-image-preview",
            refinement_iterations=2,
            output_dir=os.path.dirname(output_path),
        )

        pipeline = PaperBananaPipeline(settings=settings)

        async def run():
            return await pipeline.generate(
                GenerationInput(
                    source_context=description,
                    communicative_intent=caption or description[:200],
                    diagram_type=DiagramType.METHODOLOGY,
                )
            )

        result = asyncio.run(run())

        # Copy result to our expected output path
        if result.image_path and os.path.exists(result.image_path):
            import shutil
            shutil.copy2(result.image_path, output_path)
            return {
                "status": "success",
                "time_s": round(time.time() - t0, 1),
                "iterations": len(result.iterations),
            }

        return {"status": "error", "error": "No image generated", "time_s": round(time.time() - t0, 1)}
    except Exception as e:
        return {"status": "error", "error": str(e), "time_s": round(time.time() - t0, 1)}


def generate_matplotlib(description: str, output_path: str) -> dict:
    """Generate a matplotlib placeholder/chart."""
    t0 = time.time()
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

        ax.text(5, 4.0, "MATPLOTLIB PLACEHOLDER", ha="center", va="center",
                fontsize=14, fontweight="bold", color="#4a90d9")

        # Word-wrap description
        words = description.split()
        lines = []
        line = ""
        for w in words:
            if len(line + " " + w) > 80:
                lines.append(line)
                line = w
            else:
                line = (line + " " + w).strip()
        if line:
            lines.append(line)
        wrapped = "\n".join(lines[:6])

        ax.text(5, 2.5, wrapped, ha="center", va="center",
                fontsize=8, color="#333333", style="italic",
                fontfamily="monospace")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close()

        return {"status": "success", "time_s": round(time.time() - t0, 1), "note": "placeholder only"}
    except Exception as e:
        return {"status": "error", "error": str(e), "time_s": round(time.time() - t0, 1)}


def run_shootout(description: str, output_dir: str, caption: str = ""):
    """Run all backends and save comparison results."""
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    backends = {
        "gemini": lambda: generate_gemini(description, os.path.join(output_dir, "gemini.png")),
        "gpt": lambda: generate_gpt(description, os.path.join(output_dir, "gpt.png")),
        "paperbanana": lambda: generate_paperbanana(description, caption, os.path.join(output_dir, "paperbanana.png")),
        "matplotlib": lambda: generate_matplotlib(description, os.path.join(output_dir, "matplotlib.png")),
    }

    for name, fn in backends.items():
        print(f"\n{'='*50}")
        print(f"  Running: {name}")
        print(f"{'='*50}")
        result = fn()
        results[name] = result
        status = result["status"]
        elapsed = result.get("time_s", 0)
        if status == "success":
            print(f"  OK ({elapsed}s)")
        elif status == "skipped":
            print(f"  SKIPPED: {result.get('error', '')}")
        else:
            print(f"  FAILED: {result.get('error', '')[:100]}")

    # Save metadata
    meta = {
        "description": description,
        "caption": caption,
        "results": results,
    }
    with open(os.path.join(output_dir, "_results.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Print summary
    print(f"\n{'='*50}")
    print("  SHOOTOUT SUMMARY")
    print(f"{'='*50}")
    print(f"  Output: {output_dir}/")
    for name, r in results.items():
        icon = "OK" if r["status"] == "success" else ("--" if r["status"] == "skipped" else "XX")
        t = r.get("time_s", 0)
        print(f"  [{icon}] {name:15s} {t:>5.1f}s  {r.get('note', r.get('error', ''))[:40]}")
    print(f"\nOpen the folder to compare and pick a winner!")

    return results


def main():
    parser = argparse.ArgumentParser(description="Figure Shootout — compare all backends")
    parser.add_argument("--description", required=True, help="Detailed figure description")
    parser.add_argument("--output-dir", required=True, help="Directory for comparison outputs")
    parser.add_argument("--caption", default="", help="Short caption (used by PaperBanana)")
    args = parser.parse_args()

    load_env()
    run_shootout(args.description, args.output_dir, args.caption)


if __name__ == "__main__":
    main()