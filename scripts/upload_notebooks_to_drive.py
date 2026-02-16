#!/usr/bin/env python3
"""
Upload Colab notebooks to the Vizuara Google Drive folder using rclone.

Usage:
    python scripts/upload_notebooks_to_drive.py [--slug SLUG] [notebook_dir]

If --slug is given, uploads only notebooks from output/notebooks/{slug}/.
If no directory or slug is given, recursively uploads all .ipynb files
found under output/notebooks/.
"""

import argparse
import subprocess
import sys
from pathlib import Path

RCLONE = "/Users/raj/.local/bin/rclone"
REMOTE = "gdrive:/"
FOLDER_ID = "1ZbfI4AU_DbqeHSKkbVX28DO-Pr4_PPI6"
DRIVE_FOLDER_URL = f"https://drive.google.com/drive/folders/{FOLDER_ID}"


def upload_file(file_path: Path):
    """Upload a single file to Google Drive via rclone."""
    result = subprocess.run(
        [RCLONE, "copy", str(file_path), REMOTE,
         "--drive-root-folder-id", FOLDER_ID, "-v"],
        capture_output=True, text=True,
    )

    if result.returncode == 0:
        print(f"  Uploaded: {file_path.name}")
        return True
    else:
        print(f"  FAILED: {file_path.name} — {result.stderr.strip()}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Upload notebooks to Google Drive")
    parser.add_argument("--slug", help="Article slug — uploads only output/notebooks/{slug}/")
    parser.add_argument("notebook_dir", nargs="?", help="Directory to upload from (default: output/notebooks/)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    if args.slug:
        notebook_dir = project_root / "output" / "notebooks" / args.slug
    elif args.notebook_dir:
        notebook_dir = Path(args.notebook_dir)
        if not notebook_dir.is_absolute():
            notebook_dir = project_root / notebook_dir
    else:
        notebook_dir = project_root / "output" / "notebooks"

    if not notebook_dir.exists():
        print(f"Error: Directory not found: {notebook_dir}")
        sys.exit(1)

    # Recursively find all .ipynb files (supports per-slug subfolders)
    notebooks = sorted(notebook_dir.rglob("*.ipynb"))

    # Also pick up case study notebooks when --slug is used
    if args.slug:
        cs_nb = project_root / "output" / "case-studies" / args.slug / "case_study_notebook.ipynb"
        if cs_nb.exists():
            notebooks.append(cs_nb)

    if not notebooks:
        print(f"No .ipynb files found in {notebook_dir}")
        sys.exit(1)

    print(f"Uploading {len(notebooks)} notebooks to Google Drive...")
    print(f"Target folder: {DRIVE_FOLDER_URL}\n")

    success = 0
    for nb in notebooks:
        if upload_file(nb):
            success += 1

    print(f"\nDone: {success}/{len(notebooks)} uploaded successfully.")
    print(f"Drive folder: {DRIVE_FOLDER_URL}")


if __name__ == "__main__":
    main()
