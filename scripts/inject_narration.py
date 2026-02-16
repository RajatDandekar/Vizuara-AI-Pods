#!/usr/bin/env python3
"""
Inject narration audio player cells into a Colab notebook.

Usage:
    python scripts/inject_narration.py <drive_file_id> \
        [--notebook output/notebooks/01_world_models_first_principles.ipynb] \
        [--narration output/narration/narration_script.json]

Reads narration_script.json for segment-to-cell mappings, inserts audio
player code cells at the correct positions in the notebook. Each audio cell
uses Colab's form view (#@title) so it appears collapsed by default.

Positioning strategy:
  1. Primary: match "insert_before" text against notebook cell sources
  2. Fallback: use min(cell_indices) from the narration script
"""

import argparse
import json
import sys
from pathlib import Path


def make_label(segment_id: str) -> str:
    """Generate a human-readable label from a segment_id."""
    # Strip notebook prefix (e.g. "01_00_intro" -> "00_intro")
    parts = segment_id.split("_")
    # If first two parts are digits (e.g. "01_00_intro"), skip the first
    if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit():
        label_parts = parts[2:]
    elif len(parts) >= 2 and parts[0].isdigit():
        label_parts = parts[1:]
    else:
        label_parts = parts
    return " ".join(label_parts).replace("_", " ").title()


def make_download_cell(drive_file_id: str, intro_segment_id: str = "00_intro") -> dict:
    """Create the combined download + intro audio cell (first cell in notebook)."""
    source = [
        "#@title \U0001f3a7 Download Narration Audio & Play Introduction\n",
        "import os as _os\n",
        'if not _os.path.exists("/content/narration"):\n',
        "    !pip install -q gdown\n",
        "    import gdown\n",
        f'    gdown.download(id="{drive_file_id}", output="/content/narration.zip", quiet=False)\n',
        "    !unzip -q /content/narration.zip -d /content/narration\n",
        "    !rm /content/narration.zip\n",
        "    print(f\"Loaded {len(_os.listdir('/content/narration'))} narration segments\")\n",
        "else:\n",
        '    print("Narration audio already loaded.")\n',
        "\n",
        "from IPython.display import Audio, display\n",
        f'display(Audio("/content/narration/{intro_segment_id}.mp3"))'
    ]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": ["narration"], "cellView": "form"},
        "outputs": [],
        "source": source,
        "id": "narration_download",
    }


def make_audio_cell(segment_id: str) -> dict:
    """Create an audio player cell for a segment."""
    label = make_label(segment_id)
    source = [
        f"#@title \U0001f3a7 Listen: {label}\n",
        "from IPython.display import Audio, display\n",
        "import os as _os\n",
        f'_f = "/content/narration/{segment_id}.mp3"\n',
        "if _os.path.exists(_f):\n",
        "    display(Audio(_f))\n",
        "else:\n",
        '    print("Run the first cell to download narration audio.")'
    ]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": ["narration"], "cellView": "form"},
        "outputs": [],
        "source": source,
        "id": f"narration_{segment_id}",
    }


def get_cell_source(cell: dict) -> str:
    """Extract source text from a notebook cell."""
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return source


def find_cell_by_text(cells: list, text: str) -> int | None:
    """Find the index of the first cell whose source contains the given text."""
    for i, cell in enumerate(cells):
        if text in get_cell_source(cell):
            return i
    return None


def inject(notebook_path: str, narration_path: str, drive_file_id: str):
    with open(notebook_path) as f:
        nb = json.load(f)
    with open(narration_path) as f:
        segments = json.load(f)

    original_count = len(nb["cells"])

    # Remove any previously injected narration cells (idempotency)
    nb["cells"] = [
        c for c in nb["cells"]
        if "narration" not in c.get("metadata", {}).get("tags", [])
    ]
    clean_count = len(nb["cells"])

    # Build insertion list: (position, segment_id)
    # Skip intro segment — it's handled by the download cell
    intro_segment_id = segments[0]["segment_id"] if segments else "00_intro"
    insertions = []
    for seg in segments:
        sid = seg["segment_id"]
        if sid == intro_segment_id:
            continue

        pos = None

        # Primary: use insert_before text to find the correct cell
        insert_before = seg.get("insert_before", "")
        if insert_before:
            pos = find_cell_by_text(nb["cells"], insert_before)
            if pos is not None:
                print(f"  {sid}: matched insert_before text at cell {pos}")
            else:
                print(f"  {sid}: WARNING — insert_before text not found: "
                      f"{insert_before[:60]!r}...")

        # Fallback: use min(cell_indices)
        if pos is None and seg.get("cell_indices"):
            pos = min(seg["cell_indices"])
            # Clamp to valid range
            pos = min(pos, clean_count)
            print(f"  {sid}: falling back to cell_indices, pos={pos}")

        if pos is None:
            pos = 0
            print(f"  {sid}: WARNING — no positioning info, defaulting to 0")

        insertions.append((pos, sid))

    # Sort descending by (position, segment_id) so we insert bottom-to-top.
    # Within same position, higher segment_id inserted first → lower ends up on top.
    insertions.sort(key=lambda x: (x[0], x[1]), reverse=True)

    # Insert audio cells bottom-to-top
    for pos, sid in insertions:
        cell = make_audio_cell(sid)
        nb["cells"].insert(pos, cell)

    # Insert download+intro cell at position 0
    nb["cells"].insert(0, make_download_cell(drive_file_id, intro_segment_id))

    # Save
    with open(notebook_path, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    final_count = len(nb["cells"])
    print(f"\nNotebook: {notebook_path}")
    print(f"Injected {len(insertions) + 1} narration cells "
          f"(1 download + {len(insertions)} audio)")
    print(f"Total cells: {final_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Inject narration audio cells into a Colab notebook"
    )
    parser.add_argument(
        "drive_file_id",
        help="Google Drive file ID for narration_segments.zip"
    )
    parser.add_argument(
        "--notebook",
        default="output/notebooks/01_world_models_first_principles.ipynb",
        help="Path to the .ipynb notebook"
    )
    parser.add_argument(
        "--narration",
        default="output/narration/narration_script.json",
        help="Path to narration_script.json"
    )
    args = parser.parse_args()
    inject(args.notebook, args.narration, args.drive_file_id)


if __name__ == "__main__":
    main()