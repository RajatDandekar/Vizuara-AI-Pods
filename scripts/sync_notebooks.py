#!/usr/bin/env python3
"""
Sync notebooks to Google Drive.

Uploads teaching notebooks and case study notebooks from the local repo
to Google Drive, preserving file IDs so Colab URLs remain stable.

Usage:
    python scripts/sync_notebooks.py          # Sync all notebooks
    python scripts/sync_notebooks.py --dry-run # Preview what would be synced
"""

import subprocess
import sys

REMOTE = "gdrive:vizuara-notebooks"
NOTEBOOKS_SRC = "public/notebooks/"
NOTEBOOKS_DST = f"{REMOTE}/notebooks/"
CASESTUDIES_SRC = "public/case-studies/"
CASESTUDIES_DST = f"{REMOTE}/case-studies/"


def run_rclone(src, dst, dry_run=False):
    """Run rclone copy with appropriate flags."""
    cmd = [
        "rclone", "copy", src, dst,
        "--include", "*.ipynb",
        "--update",  # Skip files that are newer on the destination
        "-v",
        "--stats", "5s",
    ]
    if dry_run:
        cmd.append("--dry-run")

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Syncing {src} -> {dst}")
    print(f"  Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    dry_run = "--dry-run" in sys.argv

    print("=" * 60)
    print("Vizuara Notebook Sync to Google Drive")
    print("=" * 60)

    # Sync teaching notebooks
    print("\n📓 Syncing teaching notebooks...")
    rc1 = run_rclone(NOTEBOOKS_SRC, NOTEBOOKS_DST, dry_run)

    # Sync case study notebooks
    print("\n📋 Syncing case study notebooks...")
    rc2 = run_rclone(CASESTUDIES_SRC, CASESTUDIES_DST, dry_run)

    if rc1 == 0 and rc2 == 0:
        print("\n✅ All notebooks synced successfully!")
        if not dry_run:
            print("   Colab URLs remain the same (file IDs preserved).")
    else:
        print("\n❌ Some errors occurred during sync.")
        sys.exit(1)


if __name__ == "__main__":
    main()
