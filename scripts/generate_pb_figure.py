#!/usr/bin/env python3
"""Generate a figure using PaperBanana pipeline."""
import asyncio
import argparse
import shutil
from pathlib import Path


async def generate_figure(description: str, style: str, output_path: str):
    from paperbanana import PaperBananaPipeline, GenerationInput, DiagramType
    from paperbanana.core.config import Settings

    settings = Settings(
        vlm_model="gemini-2.0-flash",
        image_model="gemini-3-pro-image-preview",
        refinement_iterations=2,
    )

    pipeline = PaperBananaPipeline(settings=settings)
    result = await pipeline.generate(
        GenerationInput(
            source_context=description,
            communicative_intent=style,
            diagram_type=DiagramType.METHODOLOGY,
        )
    )

    # Copy from pipeline output to desired location
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    src = Path(result.image_path)
    shutil.copy2(str(src), str(output))
    print(f"OK: saved {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--description", required=True, help="Figure description/context")
    parser.add_argument("--style", required=True, help="Communicative intent")
    parser.add_argument("--output", required=True, help="Output PNG path")
    args = parser.parse_args()

    asyncio.run(generate_figure(args.description, args.style, args.output))


if __name__ == "__main__":
    main()
