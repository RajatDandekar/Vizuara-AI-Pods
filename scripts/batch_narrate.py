#!/usr/bin/env python3
"""
Batch narration pipeline for all Vizuara course notebooks.

Processes every teaching notebook end-to-end:
  1. Build inventory from pod.json files
  2. Parse each notebook into cells JSON
  3. Generate narration scripts via Gemini API
  4. Generate audio segments via ElevenLabs REST API
  5. Combine audio, generate timestamps, zip segments
  6. Upload segment zips to Google Drive
  7. Inject narration audio players into notebooks
  8. Copy modified notebooks to public/ for Drive sync

Usage:
    python scripts/batch_narrate.py                    # Full run
    python scripts/batch_narrate.py --resume           # Resume from state file
    python scripts/batch_narrate.py --dry-run          # Preview inventory only
    python scripts/batch_narrate.py --course diffusion-models  # Single course
    python scripts/batch_narrate.py --pod intro-ddpm   # Single pod
    python scripts/batch_narrate.py --step scripts     # Only generate scripts
    python scripts/batch_narrate.py --step audio       # Only generate audio
    python scripts/batch_narrate.py --step inject      # Only inject + upload
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
COURSE_CREATOR_DIR = "/Users/raj/Desktop/Course_Creator"
CONTENT_DIR = os.path.join(COURSE_CREATOR_DIR, "content/courses")
PUBLIC_NOTEBOOKS_DIR = os.path.join(COURSE_CREATOR_DIR, "public/notebooks")
OUTPUT_DIR = os.path.join(COURSE_CREATOR_DIR, "output/narration")
SKILLS_DIR = "/Users/raj/.claude/skills/narrate-notebook/scripts"
DRIVE_FOLDER_ID = "1ZbfI4AU_DbqeHSKkbVX28DO-Pr4_PPI6"
RCLONE_PATH = "/Users/raj/.local/bin/rclone"
STATE_FILE = os.path.join(OUTPUT_DIR, "batch_state.json")
NARRATION_GUIDE = os.path.join(COURSE_CREATOR_DIR, "narration-guide.md")
STYLE_PROFILE = os.path.join(COURSE_CREATOR_DIR, "style-profile.md")

# Add skills scripts to path for imports
sys.path.insert(0, SKILLS_DIR)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
def load_env():
    """Load environment variables from .env.local."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("ERROR: pip install python-dotenv", file=sys.stderr)
        sys.exit(1)

    env_path = os.path.join(COURSE_CREATOR_DIR, ".env.local")
    if os.path.exists(env_path):
        load_dotenv(env_path)

    required = ["GOOGLE_API_KEY", "ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        print(f"ERROR: Missing env vars: {', '.join(missing)}", file=sys.stderr)
        print(f"Add them to {env_path}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# State management (resume support)
# ---------------------------------------------------------------------------
def load_state():
    """Load progress state from disk."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"completed": {}, "failed": {}, "drive_ids": {}}


def save_state(state):
    """Persist progress state to disk."""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def notebook_key(pod_slug, nn):
    """Unique key for a notebook in the state file."""
    return f"{pod_slug}/{nn}"


# ---------------------------------------------------------------------------
# Step 1: Build inventory from pod.json files
# ---------------------------------------------------------------------------
def build_inventory(filter_course=None, filter_pod=None):
    """
    Walk content/courses/*/pods/*/pod.json and collect all teaching notebooks.
    Returns a list of dicts with all metadata needed for processing.
    """
    inventory = []

    # Find all pod.json files under the pod hierarchy
    pattern = os.path.join(CONTENT_DIR, "*/pods/*/pod.json")
    pod_files = sorted(glob.glob(pattern))

    for pod_path in pod_files:
        with open(pod_path, "r") as f:
            pod = json.load(f)

        course_slug = pod.get("courseSlug", "")
        pod_slug = pod.get("slug", "")

        if filter_course and course_slug != filter_course:
            continue
        if filter_pod and pod_slug != filter_pod:
            continue

        for nb in pod.get("notebooks", []):
            download_path = nb.get("downloadPath", "")
            if not download_path:
                continue

            # Derive filesystem path: downloadPath = /notebooks/{podSlug}/{filename}
            filename = os.path.basename(download_path)
            nn_match = re.match(r"^(\d+)_", filename)
            nn = nn_match.group(1) if nn_match else "00"
            stem = os.path.splitext(filename)[0]

            # The notebook could be at several paths. Find the right one.
            # Primary: public/notebooks/{courseSlug}/{podSlug}/{filename}
            # Fallback: public/notebooks/{podSlug}/{filename}
            candidates = [
                os.path.join(PUBLIC_NOTEBOOKS_DIR, course_slug, pod_slug, filename),
                os.path.join(PUBLIC_NOTEBOOKS_DIR, pod_slug, filename),
            ]
            local_path = None
            for c in candidates:
                if os.path.exists(c):
                    local_path = c
                    break

            if not local_path:
                print(f"  WARNING: Notebook not found: {download_path}")
                print(f"    Tried: {candidates}")
                continue

            inventory.append({
                "course_slug": course_slug,
                "pod_slug": pod_slug,
                "pod_json_path": pod_path,
                "nb_slug": nb.get("slug", ""),
                "nb_title": nb.get("title", ""),
                "nn": nn,
                "stem": stem,
                "filename": filename,
                "local_path": local_path,
                "download_path": download_path,
                "colab_url": nb.get("colabUrl", ""),
                "has_narration": nb.get("hasNarration", False),
            })

    return inventory


# ---------------------------------------------------------------------------
# Step 2: Parse notebook
# ---------------------------------------------------------------------------
def parse_notebook(nb_info):
    """Parse a notebook into cells JSON using the existing script."""
    narr_dir = os.path.join(OUTPUT_DIR, nb_info["pod_slug"])
    os.makedirs(narr_dir, exist_ok=True)

    output_json = os.path.join(narr_dir, f"{nb_info['nn']}_{nb_info['stem']}_cells.json")

    if os.path.exists(output_json):
        print(f"    [parse] Using cached: {output_json}")
        return output_json

    cmd = [
        sys.executable,
        os.path.join(SKILLS_DIR, "parse_notebook.py"),
        "--notebook", nb_info["local_path"],
        "--output", output_json,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=COURSE_CREATOR_DIR)
    if result.returncode != 0:
        print(f"    [parse] ERROR: {result.stderr}")
        return None

    print(f"    [parse] Done: {output_json}")
    return output_json


# ---------------------------------------------------------------------------
# Step 3: Generate narration script via Gemini
# ---------------------------------------------------------------------------
def _load_text_file(path):
    with open(path, "r") as f:
        return f.read()


def _build_gemini_prompt(cells_json_path, nb_info):
    """Build the prompt for Gemini to generate a narration script."""
    narration_guide = _load_text_file(NARRATION_GUIDE)

    with open(cells_json_path, "r") as f:
        cells_data = json.load(f)

    nn = nb_info["nn"]

    prompt = f"""You are a narration script writer for Vizuara, an AI education platform.
Your job is to write a conversational narration script for a Google Colab teaching notebook.

## NARRATION STYLE GUIDE
{narration_guide}

## NOTEBOOK INFORMATION
- Title: {nb_info['nb_title']}
- Course: {nb_info['course_slug']}
- Pod: {nb_info['pod_slug']}
- Notebook number: {nn}

## NOTEBOOK CELLS
{json.dumps(cells_data, indent=2)}

## YOUR TASK

Write a complete narration script as a JSON array. Each element is a segment:

```json
[
  {{
    "segment_id": "{nn}_00_intro",
    "cell_indices": [],
    "insert_before": "",
    "type": "intro",
    "narration_text": "...",
    "duration_estimate_seconds": 30
  }},
  {{
    "segment_id": "{nn}_01_section_name",
    "cell_indices": [1, 2],
    "insert_before": "## Section Heading Text",
    "type": "explanation",
    "narration_text": "...",
    "duration_estimate_seconds": 45
  }}
]
```

## RULES

1. **segment_id**: Always prefix with `{nn}_` (e.g., `{nn}_00_intro`, `{nn}_01_motivation`)
2. **insert_before**: Copy a heading or first distinctive line from the target cell EXACTLY.
   The narration audio player will be placed immediately BEFORE the cell containing this text.
   For intro segments, leave as empty string.
3. **type**: One of: intro, explanation, code_walkthrough, todo, visualization, transition, closing
4. **narration_text**: Conversational, warm, spoken language. Use contractions. Say "we" and "you".
   Do NOT read markdown verbatim. Do NOT read code syntax. Explain concepts in plain English.
   Keep each segment under 2 minutes of spoken text (~250-300 words max per segment).
5. **Duration**: Estimate based on ~150 words per minute
6. Cover ALL major sections of the notebook. Don't skip sections.
7. Always include an intro segment and a closing segment.
8. For TODO cells: encourage the student, give hints, tell them to pause the audio.
9. For code cells: explain what the code does and why, highlight key lines.
10. For visualizations: build anticipation for what they'll see.
11. Keep total narration between 8-20 minutes depending on notebook length.

## OUTPUT FORMAT
Return ONLY the JSON array. No markdown code fences. No explanation before or after.
Just the raw JSON array starting with [ and ending with ].
"""
    return prompt


def _fix_json(text):
    """Attempt to fix common JSON issues from LLM output."""
    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    # Fix single quotes to double quotes (but not inside strings)
    # This is a simple heuristic — handles most LLM output
    text = re.sub(r"(?<!\\)'([^']*?)'(?=\s*[,:\]}])", r'"\1"', text)
    # Remove any trailing text after the last ]
    last_bracket = text.rfind("]")
    if last_bracket != -1:
        text = text[:last_bracket + 1]
    return text


def generate_narration_script(nb_info, cells_json_path):
    """Generate a narration script using Gemini API."""
    narr_dir = os.path.join(OUTPUT_DIR, nb_info["pod_slug"])
    script_path = os.path.join(narr_dir, f"{nb_info['nn']}_narration_script.json")

    if os.path.exists(script_path):
        # Validate cached script has required fields
        try:
            with open(script_path, "r") as f:
                cached = json.load(f)
            if isinstance(cached, list) and len(cached) > 0:
                missing = sum(1 for s in cached if "narration_text" not in s)
                if missing == 0:
                    print(f"    [script] Using cached: {script_path}")
                    return script_path
                else:
                    print(f"    [script] Cached script has {missing} segments without narration_text, regenerating...")
                    os.remove(script_path)
        except (json.JSONDecodeError, KeyError):
            os.remove(script_path)

    prompt = _build_gemini_prompt(cells_json_path, nb_info)

    from google import genai

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    for attempt in range(3):
        try:
            config = genai.types.GenerateContentConfig(
                temperature=0.7 if attempt == 0 else 0.3,
                max_output_tokens=16000,
            )
            if attempt > 0:
                config = genai.types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=16000,
                    response_mime_type="application/json",
                )

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt if attempt == 0 else prompt + "\n\nIMPORTANT: Return ONLY valid JSON. No trailing commas. No comments. No single quotes. Ensure all strings use double quotes and are properly escaped.",
                config=config,
            )

            text = response.text.strip()

            # Strip markdown code fences if present
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*\n?", "", text)
                text = re.sub(r"\n?```\s*$", "", text)

            # Try parsing as-is first
            try:
                script = json.loads(text)
            except json.JSONDecodeError:
                # Attempt JSON fix
                text = _fix_json(text)
                script = json.loads(text)

            if not isinstance(script, list):
                raise ValueError("Expected JSON array")

            with open(script_path, "w") as f:
                json.dump(script, f, indent=2)

            print(f"    [script] Generated {len(script)} segments -> {script_path}")
            return script_path

        except Exception as e:
            print(f"    [script] Attempt {attempt+1}/3 failed: {e}")
            if attempt < 2:
                time.sleep(5)

    print(f"    [script] FAILED after 3 attempts")
    return None


# ---------------------------------------------------------------------------
# Step 4: Generate audio via ElevenLabs REST API
# ---------------------------------------------------------------------------
def generate_audio_segment(segment, segments_dir):
    """Generate a single audio segment via ElevenLabs REST API."""
    import requests

    seg_id = segment["segment_id"]
    output_path = os.path.join(segments_dir, f"{seg_id}.mp3")

    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        return output_path  # Already generated

    api_key = os.environ["ELEVENLABS_API_KEY"]
    voice_id = os.environ["ELEVENLABS_VOICE_ID"]

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key,
    }
    payload = {
        "text": segment["narration_text"],
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.8,
            "style": 0.3,
            "use_speaker_boost": True,
        },
    }

    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=120)
            if resp.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(resp.content)
                return output_path
            elif resp.status_code == 429:
                # Rate limited — wait and retry
                wait = 30 * (attempt + 1)
                print(f"      Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"      ElevenLabs error {resp.status_code}: {resp.text[:200]}")
                if attempt < 2:
                    time.sleep(5)
        except Exception as e:
            print(f"      Request error: {e}")
            if attempt < 2:
                time.sleep(5)

    return None


def generate_all_audio(nb_info, script_path):
    """Generate audio for all segments in a narration script."""
    narr_dir = os.path.join(OUTPUT_DIR, nb_info["pod_slug"])
    segments_dir = os.path.join(narr_dir, "segments")
    os.makedirs(segments_dir, exist_ok=True)

    with open(script_path, "r") as f:
        script = json.load(f)

    total = len(script)
    success = 0
    skipped = 0
    total_chars = 0

    for i, segment in enumerate(script):
        seg_id = segment.get("segment_id", f"unknown_{i}")
        text = segment.get("narration_text", segment.get("text", ""))
        if not text:
            print(f"      [{i+1}/{total}] {seg_id} - SKIP (no narration text)")
            skipped += 1
            continue
        # Ensure narration_text key exists for generate_audio_segment
        segment["narration_text"] = text
        total_chars += len(text)
        print(f"      [{i+1}/{total}] {seg_id} ({len(text)} chars)...", end=" ", flush=True)

        result = generate_audio_segment(segment, segments_dir)
        if result:
            print("OK")
            success += 1
        else:
            print("FAILED")

        # Small delay between requests to avoid rate limiting
        if i < total - 1:
            time.sleep(1)

    expected = total - skipped
    print(f"    [audio] {success}/{expected} segments generated ({skipped} skipped, {total_chars} chars total)")
    return success == expected, total_chars


# ---------------------------------------------------------------------------
# Step 5: Combine audio, timestamps
# ---------------------------------------------------------------------------
def combine_and_timestamp(nb_info):
    """Combine audio segments and generate timestamps."""
    narr_dir = os.path.join(OUTPUT_DIR, nb_info["pod_slug"])
    nn = nb_info["nn"]
    segments_dir = os.path.join(narr_dir, "segments")
    script_path = os.path.join(narr_dir, f"{nn}_narration_script.json")

    # Filter segments for this notebook only
    nb_segments = sorted(glob.glob(os.path.join(segments_dir, f"{nn}_*.mp3")))
    if not nb_segments:
        print(f"    [combine] No segments found for notebook {nn}")
        return False

    # Combine audio
    combined_path = os.path.join(narr_dir, f"{nn}_full_narration.mp3")
    cmd = [
        sys.executable,
        os.path.join(SKILLS_DIR, "combine_audio.py"),
        "--segments-dir", segments_dir,
        "--output", combined_path,
        "--pause-between", "1.5",
    ]

    # To filter only this notebook's segments, we create a temp dir
    import tempfile
    import shutil
    with tempfile.TemporaryDirectory() as tmpdir:
        for seg in nb_segments:
            shutil.copy2(seg, tmpdir)

        cmd_filtered = [
            sys.executable,
            os.path.join(SKILLS_DIR, "combine_audio.py"),
            "--segments-dir", tmpdir,
            "--output", combined_path,
            "--pause-between", "1.5",
        ]
        result = subprocess.run(cmd_filtered, capture_output=True, text=True,
                                cwd=COURSE_CREATOR_DIR)
        if result.returncode != 0:
            print(f"    [combine] ERROR: {result.stderr}")
            # Non-fatal — continue without combined file

    # Generate timestamps
    ts_path = os.path.join(narr_dir, f"{nn}_timestamps.md")
    cmd_ts = [
        sys.executable,
        os.path.join(SKILLS_DIR, "generate_timestamps.py"),
        "--script", script_path,
        "--segments-dir", segments_dir,
        "--output", ts_path,
    ]
    subprocess.run(cmd_ts, capture_output=True, text=True, cwd=COURSE_CREATOR_DIR)

    print(f"    [combine] Done")
    return True


# ---------------------------------------------------------------------------
# Step 6: Zip segments and upload to Drive
# ---------------------------------------------------------------------------
def zip_and_upload(nb_info, state):
    """Zip this notebook's segments and upload to Google Drive."""
    narr_dir = os.path.join(OUTPUT_DIR, nb_info["pod_slug"])
    nn = nb_info["nn"]
    segments_dir = os.path.join(narr_dir, "segments")
    key = notebook_key(nb_info["pod_slug"], nn)

    # Check if we already have a drive ID for this zip
    if key in state.get("drive_ids", {}):
        print(f"    [upload] Using cached drive ID: {state['drive_ids'][key]}")
        return state["drive_ids"][key]

    # Create zip of this notebook's segments (include pod_slug for uniqueness)
    zip_name = f"{nb_info['pod_slug']}_nb{nn}_segments.zip"
    zip_path = os.path.join(narr_dir, zip_name)

    nb_segments = sorted(glob.glob(os.path.join(segments_dir, f"{nn}_*.mp3")))
    if not nb_segments:
        print(f"    [upload] No segments to zip for notebook {nn}")
        return None

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for seg in nb_segments:
            zf.write(seg, os.path.basename(seg))
    print(f"    [upload] Zipped {len(nb_segments)} segments -> {zip_path}")

    # Upload to Google Drive
    cmd_upload = [
        RCLONE_PATH, "copy", zip_path, "gdrive:/",
        "--drive-root-folder-id", DRIVE_FOLDER_ID,
        "-v",
    ]
    result = subprocess.run(cmd_upload, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    [upload] ERROR uploading: {result.stderr[:200]}")
        return None

    # Get the Drive file ID
    cmd_id = [
        RCLONE_PATH, "lsjson", "gdrive:/",
        "--drive-root-folder-id", DRIVE_FOLDER_ID,
        "--no-modtime", "--no-mimetype",
    ]
    result = subprocess.run(cmd_id, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    [upload] ERROR getting ID: {result.stderr[:200]}")
        return None

    try:
        files = json.loads(result.stdout)
        drive_id = None
        for f in files:
            if f.get("Path") == zip_name:
                drive_id = f.get("ID")
                break

        if drive_id:
            state.setdefault("drive_ids", {})[key] = drive_id
            save_state(state)
            print(f"    [upload] Drive ID: {drive_id}")
            return drive_id
        else:
            print(f"    [upload] WARNING: Could not find {zip_name} on Drive")
            return None
    except json.JSONDecodeError:
        print(f"    [upload] ERROR parsing rclone output")
        return None


# ---------------------------------------------------------------------------
# Step 7: Inject narration into notebook
# ---------------------------------------------------------------------------
def inject_narration(nb_info, drive_id):
    """Inject narration audio players into the notebook."""
    narr_dir = os.path.join(OUTPUT_DIR, nb_info["pod_slug"])
    nn = nb_info["nn"]
    script_path = os.path.join(narr_dir, f"{nn}_narration_script.json")
    notebook_path = nb_info["local_path"]

    cmd = [
        sys.executable,
        os.path.join(SKILLS_DIR, "inject_narration.py"),
        "--notebook", notebook_path,
        "--script", script_path,
        "--drive-id", drive_id,
        "--output", notebook_path,  # Modify in place
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=COURSE_CREATOR_DIR)
    if result.returncode != 0:
        print(f"    [inject] ERROR: {result.stderr}")
        return False

    print(f"    [inject] {result.stdout.strip()}")
    return True


# ---------------------------------------------------------------------------
# Step 8: Update pod.json hasNarration flag
# ---------------------------------------------------------------------------
def update_pod_json(nb_info):
    """Set hasNarration: true for this notebook in pod.json."""
    pod_path = nb_info["pod_json_path"]

    with open(pod_path, "r") as f:
        pod = json.load(f)

    updated = False
    for nb in pod.get("notebooks", []):
        if nb.get("slug") == nb_info["nb_slug"]:
            if not nb.get("hasNarration", False):
                nb["hasNarration"] = True
                updated = True
            break

    if updated:
        with open(pod_path, "w") as f:
            json.dump(pod, f, indent=2, ensure_ascii=False)
        # Add trailing newline
        with open(pod_path, "a") as f:
            f.write("\n")
        print(f"    [pod.json] Updated hasNarration=true")

    return True


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def process_notebook(nb_info, state, steps="all"):
    """Process a single notebook through the full pipeline."""
    key = notebook_key(nb_info["pod_slug"], nb_info["nn"])
    label = f"{nb_info['course_slug']}/{nb_info['pod_slug']}/{nb_info['filename']}"

    # Check if already completed
    if key in state.get("completed", {}) and steps == "all":
        print(f"  SKIP (completed): {label}")
        return True

    print(f"\n  Processing: {label}")

    # Step 2: Parse
    if steps in ("all", "scripts"):
        cells_json = parse_notebook(nb_info)
        if not cells_json:
            state.setdefault("failed", {})[key] = "parse_failed"
            save_state(state)
            return False

    # Step 3: Generate narration script
    if steps in ("all", "scripts"):
        cells_json = os.path.join(
            OUTPUT_DIR, nb_info["pod_slug"],
            f"{nb_info['nn']}_{nb_info['stem']}_cells.json"
        )
        script_path = generate_narration_script(nb_info, cells_json)
        if not script_path:
            state.setdefault("failed", {})[key] = "script_failed"
            save_state(state)
            return False

        if steps == "scripts":
            return True

    # Step 4: Generate audio
    if steps in ("all", "audio"):
        script_path = os.path.join(
            OUTPUT_DIR, nb_info["pod_slug"],
            f"{nb_info['nn']}_narration_script.json"
        )
        if not os.path.exists(script_path):
            print(f"    [audio] No script found, skipping")
            return False

        audio_ok, chars_used = generate_all_audio(nb_info, script_path)
        if not audio_ok:
            state.setdefault("failed", {})[key] = "audio_failed"
            save_state(state)
            return False

        if steps == "audio":
            return True

    # Step 5: Combine and timestamp
    if steps in ("all", "inject"):
        combine_and_timestamp(nb_info)

    # Step 6: Zip and upload
    if steps in ("all", "inject"):
        drive_id = zip_and_upload(nb_info, state)
        if not drive_id:
            state.setdefault("failed", {})[key] = "upload_failed"
            save_state(state)
            return False

    # Step 7: Inject narration
    if steps in ("all", "inject"):
        drive_id = state.get("drive_ids", {}).get(key)
        if not drive_id:
            print(f"    [inject] No drive ID found, skipping")
            return False

        if not inject_narration(nb_info, drive_id):
            state.setdefault("failed", {})[key] = "inject_failed"
            save_state(state)
            return False

    # Step 8: Update pod.json
    if steps in ("all", "inject"):
        update_pod_json(nb_info)

    # Mark completed
    if steps == "all":
        state.setdefault("completed", {})[key] = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "drive_id": state.get("drive_ids", {}).get(key, ""),
        }
        save_state(state)

    return True


def main():
    parser = argparse.ArgumentParser(description="Batch narration pipeline")
    parser.add_argument("--course", help="Process only this course slug")
    parser.add_argument("--pod", help="Process only this pod slug")
    parser.add_argument("--resume", action="store_true", help="Resume from state file")
    parser.add_argument("--dry-run", action="store_true", help="Preview inventory only")
    parser.add_argument(
        "--step",
        choices=["all", "scripts", "audio", "inject"],
        default="all",
        help="Run only a specific step of the pipeline",
    )
    args = parser.parse_args()

    # Load environment
    load_env()

    # Build inventory
    print("=" * 70)
    print("VIZUARA BATCH NARRATION PIPELINE")
    print("=" * 70)

    inventory = build_inventory(filter_course=args.course, filter_pod=args.pod)

    # Group by course for display
    by_course = {}
    for nb in inventory:
        by_course.setdefault(nb["course_slug"], []).append(nb)

    print(f"\nInventory: {len(inventory)} teaching notebooks across {len(by_course)} courses\n")
    for course, nbs in sorted(by_course.items()):
        pods = set(nb["pod_slug"] for nb in nbs)
        already_narrated = sum(1 for nb in nbs if nb["has_narration"])
        print(f"  {course}: {len(nbs)} notebooks in {len(pods)} pods "
              f"({already_narrated} already narrated)")

    if args.dry_run:
        print("\n[DRY RUN] Exiting without processing.")
        return

    # Load or initialize state
    state = load_state() if args.resume else {"completed": {}, "failed": {}, "drive_ids": {}}

    # Process all notebooks
    total = len(inventory)
    success = 0
    failed = 0
    skipped = 0
    total_start = time.time()

    for i, nb_info in enumerate(inventory):
        key = notebook_key(nb_info["pod_slug"], nb_info["nn"])
        print(f"\n{'='*60}")
        print(f"[{i+1}/{total}] {nb_info['course_slug']}/{nb_info['pod_slug']}/{nb_info['filename']}")
        print(f"{'='*60}")

        if key in state.get("completed", {}) and args.step == "all":
            print("  Already completed, skipping.")
            skipped += 1
            continue

        ok = process_notebook(nb_info, state, steps=args.step)
        if ok:
            success += 1
        else:
            failed += 1

    elapsed = time.time() - total_start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print(f"\n{'='*70}")
    print(f"BATCH COMPLETE")
    print(f"{'='*70}")
    print(f"  Total notebooks: {total}")
    print(f"  Successful:      {success}")
    print(f"  Failed:          {failed}")
    print(f"  Skipped:         {skipped}")
    print(f"  Time elapsed:    {minutes}m {seconds}s")
    print(f"  State saved to:  {STATE_FILE}")

    if failed > 0:
        print(f"\n  Failed notebooks:")
        for key, reason in state.get("failed", {}).items():
            print(f"    {key}: {reason}")

    print(f"\nNext steps:")
    print(f"  1. Review output in: {OUTPUT_DIR}")
    print(f"  2. Sync notebooks to Drive: python scripts/sync_notebooks.py")


if __name__ == "__main__":
    main()
