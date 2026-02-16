#!/usr/bin/env python3
"""
Upload a file to the Vizuara Google Drive folder using rclone.

Usage:
    python scripts/upload_to_drive.py <file_path>
"""

import subprocess
import sys
from pathlib import Path

RCLONE = "/Users/raj/.local/bin/rclone"
REMOTE = "gdrive:/"
FOLDER_ID = "1kvHB2fPlnbH6-SXd-SP0hoahcr8yaIER"
DRIVE_FOLDER_URL = f"https://drive.google.com/drive/folders/{FOLDER_ID}"


def upload_file(file_path: Path):
    """Upload a file to Google Drive via rclone."""
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    result = subprocess.run(
        [RCLONE, "copy", str(file_path), REMOTE,
         "--drive-root-folder-id", FOLDER_ID, "-v"],
        capture_output=True, text=True,
    )

    if result.returncode == 0:
        print(f"Uploaded: {file_path.name}")
        print(f"Drive folder: {DRIVE_FOLDER_URL}")
    else:
        print(f"Upload failed: {result.stderr}")
        sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/upload_to_drive.py <file_path>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    if not file_path.is_absolute():
        file_path = Path(__file__).resolve().parent.parent / file_path

    upload_file(file_path)


if __name__ == "__main__":
    main()