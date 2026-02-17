#!/usr/bin/env python3
"""
Inject AI Teaching Assistant link cell into Colab notebooks.

Usage:
    python scripts/inject_chatbot.py --notebook path/to/notebook.ipynb
    python scripts/inject_chatbot.py --slug world-models
    python scripts/inject_chatbot.py --all-published

The chatbot cell:
  - A simple markdown cell with a link to the web-based AI Teaching Assistant
  - Opens the assistant page side-by-side with Colab for better UX
  - Positioned right after the notebook title for maximum visibility
  - Tagged with metadata {"tags": ["chatbot"]} for idempotent re-injection
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PUBLIC_NB_DIR = PROJECT_ROOT / "public" / "notebooks"
OUTPUT_NB_DIR = PROJECT_ROOT / "output" / "notebooks"
PUBLIC_CS_DIR = PROJECT_ROOT / "public" / "case-studies"

SITE_URL = "https://course-creator-brown.vercel.app"


# ── Context extraction (kept for external use) ───────────────────────────────


def extract_context(nb: dict) -> str:
    """Extract all non-chatbot cell content as text for the AI context."""
    parts = []
    for cell in nb.get("cells", []):
        tags = cell.get("metadata", {}).get("tags", [])
        if "chatbot" in tags or "narration" in tags:
            continue
        src = "".join(cell.get("source", []))
        if src.strip():
            ct = cell.get("cell_type", "unknown")
            parts.append(f"[{ct.upper()}]\n{src}")
    context = "\n\n---\n\n".join(parts)
    return context[:80000]


# ── Cell construction ─────────────────────────────────────────────────────────


def make_chatbot_cell(slug: str, order: int, site_url: str = SITE_URL) -> dict:
    """Create a simple markdown cell linking to the web-based AI Teaching Assistant."""
    assistant_url = f"{site_url}/courses/{slug}/practice/{order}/assistant"

    source = [
        "# \U0001f916 AI Teaching Assistant\n",
        "\n",
        "Need help with this notebook? Open the **AI Teaching Assistant** — it has already read this entire notebook and can help with concepts, code, and exercises.\n",
        "\n",
        f"**[\U0001f449 Open AI Teaching Assistant]({assistant_url})**\n",
        "\n",
        "*Tip: Open it in a separate tab and work through this notebook side-by-side.*\n",
    ]

    return {
        "cell_type": "markdown",
        "metadata": {"tags": ["chatbot"]},
        "source": source,
        "id": "vizuara_chatbot",
    }


# ── Positioning ───────────────────────────────────────────────────────────────


def find_insert_position(cells: list[dict]) -> int:
    """Find the best position: right after the first title markdown cell.

    Falls back to position 1 if no title cell is found.
    """
    for i, cell in enumerate(cells):
        # Skip narration and chatbot cells
        tags = cell.get("metadata", {}).get("tags", [])
        if "narration" in tags or "chatbot" in tags:
            continue
        if cell.get("cell_type") == "markdown":
            src = "".join(cell.get("source", []))
            stripped = src.strip()
            if stripped.startswith("# ") and not stripped.startswith("## "):
                return i + 1
    return min(1, len(cells))


# ── Injection ─────────────────────────────────────────────────────────────────


def _detect_slug_and_order(notebook_path: str) -> tuple[str, int]:
    """Detect course slug and notebook order from the file path.

    Expects paths like:
      public/notebooks/{slug}/01_topic.ipynb  → (slug, 1)
      public/case-studies/{slug}/case_study_notebook.ipynb → (slug, 0)
    """
    p = Path(notebook_path)
    slug = p.parent.name
    # Try to extract order from filename prefix (e.g. "01_...", "02_...")
    match = __import__("re").match(r"^(\d+)", p.stem)
    order = int(match.group(1)) if match else 0
    return slug, order


def inject(notebook_path: str, site_url: str = SITE_URL):
    """Inject chatbot link cell into a notebook (idempotent)."""
    with open(notebook_path) as f:
        nb = json.load(f)

    original_count = len(nb["cells"])

    # Remove existing chatbot cells
    nb["cells"] = [
        c for c in nb["cells"]
        if "chatbot" not in c.get("metadata", {}).get("tags", [])
    ]
    removed = original_count - len(nb["cells"])

    # Create chatbot link cell
    slug, order = _detect_slug_and_order(notebook_path)
    chatbot = make_chatbot_cell(slug, order, site_url)

    # Insert right after the notebook title
    pos = find_insert_position(nb["cells"])
    nb["cells"].insert(pos, chatbot)

    # Save
    with open(notebook_path, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    status = f"replaced {removed}" if removed else "added"
    print(f"  [+] Chatbot {status} in {Path(notebook_path).name} (pos {pos})")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Inject AI Teaching Assistant chatbot into Colab notebooks"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--notebook",
        help="Path to a single .ipynb notebook"
    )
    group.add_argument(
        "--slug",
        help="Inject into all notebooks for a course slug (in output/ and public/)"
    )
    group.add_argument(
        "--all-published",
        action="store_true",
        help="Inject into all published notebooks in public/notebooks/"
    )
    parser.add_argument(
        "--site-url",
        default=SITE_URL,
        help=f"Site base URL (default: {SITE_URL})"
    )
    args = parser.parse_args()

    if args.notebook:
        path = Path(args.notebook)
        if not path.exists():
            print(f"Error: {path} not found")
            sys.exit(1)
        inject(str(path), args.site_url)

    elif args.slug:
        slug = args.slug
        count = 0
        for nb_dir in [OUTPUT_NB_DIR / slug, PUBLIC_NB_DIR / slug]:
            if nb_dir.exists():
                for nb in sorted(nb_dir.glob("*.ipynb")):
                    if nb.name == "00_index.ipynb":
                        continue
                    inject(str(nb), args.site_url)
                    count += 1
        # Also check case study notebooks
        cs_nb = PUBLIC_CS_DIR / slug / "case_study_notebook.ipynb"
        if cs_nb.exists():
            inject(str(cs_nb), args.site_url)
            count += 1
        print(f"\nInjected chatbot into {count} notebook(s) for '{slug}'")

    elif args.all_published:
        count = 0
        if PUBLIC_NB_DIR.exists():
            for slug_dir in sorted(PUBLIC_NB_DIR.iterdir()):
                if not slug_dir.is_dir():
                    continue
                for nb in sorted(slug_dir.glob("*.ipynb")):
                    if nb.name == "00_index.ipynb":
                        continue
                    inject(str(nb), args.site_url)
                    count += 1
        # Case study notebooks
        if PUBLIC_CS_DIR.exists():
            for slug_dir in sorted(PUBLIC_CS_DIR.iterdir()):
                if not slug_dir.is_dir():
                    continue
                cs_nb = slug_dir / "case_study_notebook.ipynb"
                if cs_nb.exists():
                    inject(str(cs_nb), args.site_url)
                    count += 1
        print(f"\nInjected chatbot into {count} notebook(s)")


if __name__ == "__main__":
    main()
