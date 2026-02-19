#!/usr/bin/env python3
"""
Interactive pod publisher for VIZflix (Course → Pod hierarchy).

Scans output/articles/ for completed articles, detects associated notebooks
and narration, and lets you choose which articles to publish as pods
within a parent course.

Usage:
    python scripts/publish_course.py                           # Interactive mode
    python scripts/publish_course.py --course gpu-programming --pod my-new-pod
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Allow importing sibling scripts
sys.path.insert(0, str(Path(__file__).resolve().parent))
from inject_chatbot import inject as inject_chatbot_cell

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTICLES_DIR = PROJECT_ROOT / "output" / "articles"
NOTEBOOKS_DIR = PROJECT_ROOT / "output" / "notebooks"
NARRATION_DIR = PROJECT_ROOT / "output" / "narration"
CASE_STUDIES_DIR = PROJECT_ROOT / "output" / "case-studies"
CONTENT_DIR = PROJECT_ROOT / "content" / "courses"
PUBLIC_DIR = PROJECT_ROOT / "public" / "courses"
PUBLIC_NB_DIR = PROJECT_ROOT / "public" / "notebooks"
PUBLIC_CS_DIR = PROJECT_ROOT / "public" / "case-studies"
CATALOG_PATH = CONTENT_DIR / "catalog.json"

RCLONE = "/Users/raj/.local/bin/rclone"
DRIVE_FOLDER_ID = "1ZbfI4AU_DbqeHSKkbVX28DO-Pr4_PPI6"


# ── Scanning ──────────────────────────────────────────────────────────────────


def extract_title(final_md: Path) -> str:
    """Extract title from the first # heading."""
    for line in final_md.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            return stripped[2:].strip()
    return "Untitled"


def extract_description(final_md: Path) -> str:
    """Extract subtitle/description (first substantive line after the title)."""
    lines = final_md.read_text(encoding="utf-8").splitlines()
    found_title = False
    for line in lines:
        stripped = line.strip()
        if not found_title and stripped.startswith("# ") and not stripped.startswith("## "):
            found_title = True
            continue
        if found_title and stripped:
            # Skip metadata lines
            if stripped.lower().startswith("vizuara") or stripped.startswith("---"):
                continue
            if stripped.startswith("*") and stripped.endswith("*"):
                # Italic subtitle — strip the asterisks
                return stripped.strip("*").strip()
            return stripped
    return ""


def auto_detect_tags(final_md: Path) -> list[str]:
    """Extract tags from ## headings in the article."""
    tags = []
    skip = {"conclusion", "summary", "references", "introduction", "table of contents"}
    for line in final_md.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            heading = line[3:].strip().rstrip(":")
            # Remove numbering like "1. " or "1) "
            heading = re.sub(r"^\d+[\.\)]\s*", "", heading)
            if heading.lower() not in skip and len(heading) < 60:
                tags.append(heading)
    return tags[:5]


def scan_articles() -> list[dict]:
    """Scan output/articles/ and return metadata for each publishable article."""
    if not ARTICLES_DIR.exists():
        return []

    articles = []
    for article_dir in sorted(ARTICLES_DIR.iterdir()):
        if not article_dir.is_dir():
            continue

        slug = article_dir.name
        final_md = article_dir / "final.md"
        if not final_md.exists():
            continue

        title = extract_title(final_md)
        description = extract_description(final_md)

        # Figures and equations
        fig_dir = article_dir / "figures"
        figures = sorted(fig_dir.glob("*.png")) if fig_dir.exists() else []
        eq_dir = article_dir / "equations"
        equations = sorted(eq_dir.glob("*.png")) if eq_dir.exists() else []

        # Notebooks (per-slug subfolder)
        nb_dir = NOTEBOOKS_DIR / slug
        notebooks = []
        if nb_dir.exists():
            notebooks = sorted(
                f for f in nb_dir.glob("*.ipynb")
                if f.name != "00_index.ipynb"
            )

        # Narration — check per-slug first, then notebooks
        narr_dir = NARRATION_DIR / slug
        has_narration = narr_dir.exists() and any(narr_dir.glob("*.mp3"))
        if not has_narration:
            for nb_path in notebooks:
                if _detect_notebook_narration(nb_path):
                    has_narration = True
                    break

        # Case study
        case_study = detect_case_study(slug)

        # Check if already published as a pod (look for pod.json in any course)
        already_published = _find_existing_pod(slug) is not None

        articles.append({
            "slug": slug,
            "title": title,
            "description": description,
            "dir": article_dir,
            "figures": figures,
            "equations": equations,
            "notebooks": notebooks,
            "has_narration": has_narration,
            "case_study": case_study,
            "already_published": already_published,
        })

    return articles


def _find_existing_pod(pod_slug: str) -> dict | None:
    """Find an existing pod across all courses. Returns {courseSlug, order} or None."""
    for course_dir in CONTENT_DIR.iterdir():
        if not course_dir.is_dir():
            continue
        course_json = course_dir / "course.json"
        if not course_json.exists():
            continue
        try:
            with open(course_json, encoding="utf-8") as f:
                manifest = json.load(f)
            for pod in manifest.get("pods", []):
                if pod["slug"] == pod_slug:
                    return {"courseSlug": manifest["slug"], "order": pod.get("order", 1)}
        except (json.JSONDecodeError, OSError):
            continue
    return None


# ── Case study detection ─────────────────────────────────────────────────────


def detect_case_study(slug: str) -> dict | None:
    """Detect a case study in output/case-studies/{slug}/ and extract metadata."""
    cs_dir = CASE_STUDIES_DIR / slug
    cs_md = cs_dir / "case_study.md"
    if not cs_md.exists():
        return None

    text = cs_md.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Extract title from first # heading
    title = "Case Study"
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            raw = stripped[2:].strip()
            title = re.sub(r"^Case Study:\s*", "", raw).strip()
            break

    # Extract subtitle from first ## heading
    subtitle = ""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## ") and not stripped.startswith("### "):
            subtitle = stripped[3:].strip()
            break

    # Extract company name
    company = ""
    for line in lines:
        stripped = line.strip()
        match = re.match(r"^###\s*Company Profile:\s*(.+)", stripped)
        if match:
            company = match.group(1).strip()
            break

    # Extract industry
    industry = ""
    for line in lines:
        stripped = line.strip()
        match = re.match(r"^###\s*Industry:\s*(.+)", stripped)
        if match:
            industry = match.group(1).strip()
            break

    # Extract description
    description = ""
    in_challenge = False
    for line in lines:
        stripped = line.strip()
        if "### Business Challenge" in stripped:
            in_challenge = True
            continue
        if in_challenge and stripped and not stripped.startswith("#"):
            description = stripped[:300]
            break

    has_pdf = (cs_dir / "case_study.pdf").exists()
    has_notebook = (cs_dir / "case_study_notebook.ipynb").exists()

    return {
        "dir": cs_dir,
        "title": title,
        "subtitle": subtitle,
        "company": company,
        "industry": industry,
        "description": description,
        "has_pdf": has_pdf,
        "has_notebook": has_notebook,
    }


# ── Narration detection ───────────────────────────────────────────────────────


def _detect_notebook_narration(notebook_path: Path) -> bool:
    """Check if a notebook has embedded narration cells."""
    try:
        with open(notebook_path, encoding="utf-8") as f:
            nb = json.load(f)
        for cell in nb.get("cells", []):
            tags = cell.get("metadata", {}).get("tags", [])
            if "narration" in tags:
                return True
            source = "".join(cell.get("source", []))
            if "_DRIVE_ID" in source and "_SEG" in source:
                return True
            if "narration" in source.lower() and "gdown" in source.lower():
                return True
        return False
    except (json.JSONDecodeError, OSError):
        return False


# ── Drive integration ─────────────────────────────────────────────────────────


def get_drive_notebook_ids() -> dict[str, str]:
    """Query Drive folder via rclone to get filename → file ID mapping."""
    result = subprocess.run(
        [RCLONE, "lsjson", "gdrive:/",
         "--drive-root-folder-id", DRIVE_FOLDER_ID,
         "--no-modtime", "--no-mimetype"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  Warning: Could not query Drive — {result.stderr.strip()}")
        return {}

    id_map = {}
    try:
        for f in json.loads(result.stdout):
            if f.get("Path", "").endswith(".ipynb"):
                id_map[f["Path"]] = f["ID"]
    except json.JSONDecodeError:
        pass

    return id_map


# ── Drive upload ─────────────────────────────────────────────────────────────


def _sync_notebooks_to_drive(notebooks: list[Path]):
    """Re-upload notebooks to Drive so Colab gets the latest version."""
    for nb_path in notebooks:
        result = subprocess.run(
            [RCLONE, "copy", str(nb_path), "gdrive:/",
             "--drive-root-folder-id", DRIVE_FOLDER_ID, "-v"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            if "Copied" in result.stderr:
                print(f"    ↑ Synced to Drive: {nb_path.name}")
        else:
            print(f"    [!] Drive upload failed for {nb_path.name}: {result.stderr.strip()}")


def _upload_case_study_notebook(src_path: Path, drive_name: str):
    """Upload a case study notebook to Drive with a slug-prefixed name."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        dest = Path(tmpdir) / drive_name
        shutil.copy2(src_path, dest)
        result = subprocess.run(
            [RCLONE, "copy", str(dest), "gdrive:/",
             "--drive-root-folder-id", DRIVE_FOLDER_ID, "-v"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"    ↑ Uploaded case study notebook as {drive_name}")
        else:
            print(f"    [!] Drive upload failed for {drive_name}: {result.stderr.strip()}")


# ── Notebook metadata extraction ──────────────────────────────────────────────


def build_notebook_meta(pod_slug: str, notebooks: list[Path], drive_ids: dict) -> list[dict]:
    """Build NotebookMeta entries for pod.json."""
    metas = []
    for i, nb_path in enumerate(notebooks, 1):
        try:
            with open(nb_path, encoding="utf-8") as f:
                nb = json.load(f)
        except (json.JSONDecodeError, OSError):
            print(f"    Warning: Could not parse {nb_path.name}, skipping")
            continue

        cells = nb.get("cells", [])
        title = f"Notebook {i}"
        objective = ""
        todo_count = 0

        # Extract title from first non-narration markdown heading
        for cell in cells:
            tags = cell.get("metadata", {}).get("tags", [])
            if "narration" in tags:
                continue
            if cell.get("cell_type") != "markdown":
                continue
            source = "".join(cell.get("source", []))
            for line in source.splitlines():
                line = line.strip()
                if line.startswith("# ") and not line.startswith("## "):
                    title = line[2:].strip()
                    title = re.sub(r"^[^\w]+", "", title).strip()
                    break
            if title != f"Notebook {i}":
                for line in source.splitlines()[1:]:
                    line = line.strip()
                    if line and not line.startswith("#") and not line.startswith("*"):
                        objective = line[:200]
                        break
                break

        # Count TODO markers
        for cell in cells:
            source = "".join(cell.get("source", []))
            todo_count += source.upper().count("# TODO")

        # Estimate minutes
        non_narration = [c for c in cells
                         if "narration" not in c.get("metadata", {}).get("tags", [])]
        estimated_minutes = max(30, round(len(non_narration) * 1.5))

        # Colab URL from Drive ID
        drive_id = drive_ids.get(nb_path.name, "")
        colab_url = f"https://colab.research.google.com/drive/{drive_id}" if drive_id else ""

        has_narration = _detect_notebook_narration(nb_path)

        # Derive a URL-friendly slug from filename
        stem = nb_path.stem
        parts = stem.split("_", 1)
        clean_slug = f"{parts[0]}-{parts[1]}" if len(parts) > 1 else stem

        metas.append({
            "title": title,
            "slug": clean_slug,
            "objective": objective,
            "colabUrl": colab_url,
            "downloadPath": f"/notebooks/{pod_slug}/{nb_path.name}",
            "hasNarration": has_narration,
            "estimatedMinutes": estimated_minutes,
            "todoCount": todo_count,
            "order": i,
        })

    return metas


# ── Course listing ────────────────────────────────────────────────────────────


def get_available_courses() -> list[dict]:
    """List all courses that have a course.json."""
    courses = []
    for course_dir in sorted(CONTENT_DIR.iterdir()):
        if not course_dir.is_dir():
            continue
        course_json = course_dir / "course.json"
        if not course_json.exists():
            continue
        try:
            with open(course_json, encoding="utf-8") as f:
                manifest = json.load(f)
            courses.append(manifest)
        except (json.JSONDecodeError, OSError):
            continue
    return courses


def select_course(courses: list[dict], preset: str | None = None) -> dict | None:
    """Prompt user to select a target course (or use preset slug)."""
    if preset:
        for c in courses:
            if c["slug"] == preset:
                return c
        print(f"  Error: Course '{preset}' not found.")
        return None

    print()
    print("  Available courses:")
    print(f"  {'─' * 50}")
    for i, c in enumerate(courses, 1):
        pod_count = len(c.get("pods", []))
        live_pods = sum(1 for p in c.get("pods", []) if p.get("notebookCount", 0) > 0)
        print(f"  {i:3d}. {c['title'][:45]}")
        print(f"       {c['slug']}  ({live_pods}/{pod_count} pods with content)")
    print()

    choice = input("  Select course number (or 'q' to quit): ").strip()
    if choice in ("q", "quit", ""):
        return None
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(courses):
            return courses[idx]
    except ValueError:
        pass
    print("  Invalid selection.")
    return None


# ── Publishing ────────────────────────────────────────────────────────────────


def publish_pod(article: dict, course_slug: str, drive_ids: dict):
    """Publish a single article as a pod within a parent course."""
    pod_slug = article["slug"]
    print(f"\n{'─' * 60}")
    print(f"  Publishing pod: {article['title']}")
    print(f"  Course: {course_slug} / Pod: {pod_slug}")
    print(f"{'─' * 60}")

    # 1. Create pod content directory
    pod_dir = CONTENT_DIR / course_slug / "pods" / pod_slug
    pod_dir.mkdir(parents=True, exist_ok=True)

    # 2. Copy final.md → article.md
    shutil.copy2(article["dir"] / "final.md", pod_dir / "article.md")
    print(f"  [+] Copied article.md")

    # 3. Copy figures to public/ (nested under course/pod)
    if article["figures"]:
        public_fig_dir = PUBLIC_DIR / course_slug / "pods" / pod_slug / "figures"
        public_fig_dir.mkdir(parents=True, exist_ok=True)
        for fig in article["figures"]:
            shutil.copy2(fig, public_fig_dir / fig.name)
        print(f"  [+] Copied {len(article['figures'])} figures")

    # 4. Copy equations to public/
    if article["equations"]:
        public_eq_dir = PUBLIC_DIR / course_slug / "pods" / pod_slug / "equations"
        public_eq_dir.mkdir(parents=True, exist_ok=True)
        for eq in article["equations"]:
            shutil.copy2(eq, public_eq_dir / eq.name)
        print(f"  [+] Copied {len(article['equations'])} equation images")

    # 5. Handle notebooks
    notebook_metas = []
    if article["notebooks"]:
        # Inject AI Teaching Assistant chatbot into each notebook
        print(f"  [~] Injecting AI Teaching Assistant chatbot...")
        for nb in article["notebooks"]:
            inject_chatbot_cell(str(nb))

        notebook_metas = build_notebook_meta(pod_slug, article["notebooks"], drive_ids)

        # Copy notebook files to public/ (flat by podSlug)
        public_nb_dir = PUBLIC_NB_DIR / pod_slug
        public_nb_dir.mkdir(parents=True, exist_ok=True)
        for nb in article["notebooks"]:
            shutil.copy2(nb, public_nb_dir / nb.name)
        print(f"  [+] Copied {len(article['notebooks'])} notebooks to public/")

        # Re-upload notebooks to Drive
        print(f"  [~] Syncing notebooks to Google Drive...")
        _sync_notebooks_to_drive(article["notebooks"])

        missing = [m for m in notebook_metas if not m["colabUrl"]]
        if missing:
            print(f"  [!] Warning: {len(missing)} notebooks missing Drive IDs (no Colab URL)")
            print(f"      Upload notebooks first: python scripts/upload_notebooks_to_drive.py --slug {pod_slug}")
    else:
        print(f"  [i] No notebooks found — publishing as article-only pod")

    # 6. Narration info
    if article["has_narration"]:
        print(f"  [i] Narration detected (source: output/narration/{pod_slug}/)")

    # 7. Handle case study
    case_study_meta = None
    cs = article.get("case_study")
    if cs:
        print()
        print(f"  --- Case Study Detected ---")
        print(f"  Title:    {cs['title'][:60]}")
        print(f"  Company:  {cs['company']}")
        print(f"  Industry: {cs['industry']}")
        print(f"  PDF:      {'yes' if cs['has_pdf'] else 'no'}")
        print(f"  Notebook: {'yes' if cs['has_notebook'] else 'no'}")

        # Copy files to public/ (flat by podSlug)
        public_cs_dir = PUBLIC_CS_DIR / pod_slug
        public_cs_dir.mkdir(parents=True, exist_ok=True)

        # Copy case_study.md to public/ and content/
        shutil.copy2(cs["dir"] / "case_study.md", public_cs_dir / "case_study.md")
        shutil.copy2(cs["dir"] / "case_study.md", pod_dir / "case_study.md")

        if cs["has_pdf"]:
            shutil.copy2(cs["dir"] / "case_study.pdf", public_cs_dir / "case_study.pdf")

        if cs["has_notebook"]:
            shutil.copy2(cs["dir"] / "case_study_notebook.ipynb", public_cs_dir / "case_study_notebook.ipynb")

        # Upload case study notebook to Drive
        cs_colab_url = ""
        if cs["has_notebook"]:
            cs_drive_name = f"{pod_slug}-case_study_notebook.ipynb"
            cs_src = cs["dir"] / "case_study_notebook.ipynb"
            _upload_case_study_notebook(cs_src, cs_drive_name)
            cs_drive_id = drive_ids.get(cs_drive_name, "")
            if not cs_drive_id:
                fresh_ids = get_drive_notebook_ids()
                cs_drive_id = fresh_ids.get(cs_drive_name, "")
                if cs_drive_id:
                    drive_ids[cs_drive_name] = cs_drive_id
            if cs_drive_id:
                cs_colab_url = f"https://colab.research.google.com/drive/{cs_drive_id}"
                print(f"  [+] Case study Colab URL: {cs_colab_url}")
            else:
                print(f"  [!] Warning: Could not get Drive ID for {cs_drive_name}")

        case_study_meta = {
            "title": cs["title"],
            "subtitle": cs["subtitle"],
            "company": cs["company"],
            "industry": cs["industry"],
            "description": cs["description"],
            "pdfPath": f"/case-studies/{pod_slug}/case_study.pdf" if cs["has_pdf"] else "",
            "colabUrl": cs_colab_url,
            "notebookPath": f"/case-studies/{pod_slug}/case_study_notebook.ipynb" if cs["has_notebook"] else "",
        }
        print(f"  [+] Published case study files")
    else:
        print(f"\n  [i] No case study found for this pod")

    # 8. Prompt for metadata
    print()
    difficulty = input(f"  Difficulty [beginner/intermediate/advanced] (default: intermediate): ").strip()
    if difficulty not in ("beginner", "intermediate", "advanced"):
        difficulty = "intermediate"

    word_count = len((article["dir"] / "final.md").read_text(encoding="utf-8").split())
    article_hours = word_count / 200 / 60
    nb_hours = sum(m["estimatedMinutes"] for m in notebook_metas) / 60
    default_hours = max(1, round(article_hours + nb_hours))

    est_input = input(f"  Estimated hours (default: {default_hours}): ").strip()
    est_hours = int(est_input) if est_input.isdigit() else default_hours

    tags_input = input(f"  Tags (comma-separated, or Enter to auto-detect): ").strip()
    if tags_input:
        tags = [t.strip() for t in tags_input.split(",")]
    else:
        tags = auto_detect_tags(article["dir"] / "final.md")
        print(f"  Auto-detected tags: {', '.join(tags)}")

    # 9. Build pod.json
    pod_manifest = {
        "title": article["title"],
        "slug": pod_slug,
        "description": article["description"],
        "difficulty": difficulty,
        "estimatedHours": est_hours,
        "prerequisites": [],
        "tags": tags,
        "article": {
            "notionUrl": None,
            "figureUrls": {},
        },
        "notebooks": notebook_metas,
        "curator": {
            "name": "Dr. Rajat Dandekar",
            "title": "Course Instructor",
            "bio": "Dr. Rajat Dandekar is a researcher and educator specializing in AI/ML, with a passion for making complex concepts accessible through intuitive explanations and hands-on learning.",
            "videoUrl": "https://drive.google.com/file/d/1xgSDKFZLU25MjUogCs4siGb5PJKSQGJN/view?usp=sharing",
            "imageUrl": "/founders/rajat.jpg",
        },
        "courseSlug": course_slug,
    }

    if case_study_meta:
        pod_manifest["caseStudy"] = case_study_meta

    with open(pod_dir / "pod.json", "w", encoding="utf-8") as f:
        json.dump(pod_manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"  [+] Generated pod.json")

    # 10. Update parent course.json
    update_course_manifest(course_slug, pod_slug, article["title"],
                           article["description"], est_hours,
                           len(notebook_metas), case_study_meta is not None)
    print(f"  [+] Updated course.json")

    # 11. Update catalog.json
    update_catalog_from_course(course_slug)
    print(f"  [+] Updated catalog.json")


def update_course_manifest(course_slug: str, pod_slug: str, title: str,
                           description: str, est_hours: int,
                           nb_count: int, has_case_study: bool):
    """Add or update a pod entry in the parent course.json."""
    course_json = CONTENT_DIR / course_slug / "course.json"
    with open(course_json, encoding="utf-8") as f:
        manifest = json.load(f)

    pods = manifest.get("pods", [])

    # Find existing pod entry
    existing_idx = None
    for i, p in enumerate(pods):
        if p["slug"] == pod_slug:
            existing_idx = i
            break

    # Determine thumbnail
    fig_path = f"/courses/{course_slug}/pods/{pod_slug}/figures/figure_1.png"
    fig_file = PUBLIC_DIR / course_slug / "pods" / pod_slug / "figures" / "figure_1.png"
    thumbnail = fig_path if fig_file.exists() else None

    pod_entry = {
        "slug": pod_slug,
        "title": title,
        "description": description,
        "order": existing_idx + 1 if existing_idx is not None else len(pods) + 1,
        "notebookCount": nb_count,
        "estimatedHours": est_hours,
        "hasCaseStudy": has_case_study,
    }
    if thumbnail:
        pod_entry["thumbnail"] = thumbnail

    if existing_idx is not None:
        # Preserve order from existing entry
        pod_entry["order"] = pods[existing_idx].get("order", existing_idx + 1)
        pods[existing_idx] = pod_entry
    else:
        pods.append(pod_entry)

    manifest["pods"] = pods

    # Update course-level thumbnail if not set
    if not manifest.get("thumbnail") and thumbnail:
        manifest["thumbnail"] = thumbnail

    with open(course_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")


def update_catalog_from_course(course_slug: str):
    """Re-aggregate catalog entry from course.json data."""
    course_json = CONTENT_DIR / course_slug / "course.json"
    with open(course_json, encoding="utf-8") as f:
        manifest = json.load(f)

    pods = manifest.get("pods", [])
    pod_count = len(pods)
    total_notebooks = sum(p.get("notebookCount", 0) for p in pods)
    total_hours = sum(p.get("estimatedHours", 0) for p in pods)
    # Only count pods with actual content as contributing to estimatedHours
    live_pods = [p for p in pods if p.get("notebookCount", 0) > 0]
    if live_pods:
        total_hours = sum(p.get("estimatedHours", 0) for p in live_pods)

    # Load catalog
    if CATALOG_PATH.exists():
        with open(CATALOG_PATH, encoding="utf-8") as f:
            catalog = json.load(f)
    else:
        catalog = {"courses": []}

    # Find and update existing entry
    found = False
    for entry in catalog["courses"]:
        if entry["slug"] == course_slug:
            entry["podCount"] = pod_count
            entry["totalNotebooks"] = total_notebooks
            entry["estimatedHours"] = max(1, total_hours)
            # Update thumbnail if course has one
            if manifest.get("thumbnail"):
                entry["thumbnail"] = manifest["thumbnail"]
            found = True
            break

    if not found:
        # Add new catalog entry from course manifest
        catalog["courses"].append({
            "slug": course_slug,
            "title": manifest["title"],
            "description": manifest["description"],
            "difficulty": manifest.get("difficulty", "intermediate"),
            "estimatedHours": max(1, total_hours),
            "tags": manifest.get("tags", []),
            "podCount": pod_count,
            "totalNotebooks": total_notebooks,
            "status": "live",
            "thumbnail": manifest.get("thumbnail", ""),
        })

    # Sort by title
    catalog["courses"].sort(key=lambda c: c["title"])

    with open(CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
        f.write("\n")


# ── Interactive UI ────────────────────────────────────────────────────────────


def display_menu(articles: list[dict]):
    """Show interactive menu with article status."""
    print()
    print("=" * 64)
    print("   VIZUARA POD PUBLISHER")
    print("=" * 64)
    print()

    for i, art in enumerate(articles, 1):
        status = "PUBLISHED" if art["already_published"] else "NEW"
        nb_count = len(art["notebooks"])

        info_parts = []
        info_parts.append(f"{len(art['figures'])} figures")
        if nb_count > 0:
            info_parts.append(f"{nb_count} notebooks")
        else:
            info_parts.append("article-only")
        if art["has_narration"]:
            info_parts.append("narrated")
        if art["case_study"]:
            info_parts.append("case study")

        print(f"  {i:2d}. [{status:>9s}]  {art['title'][:48]}")
        print(f"      {art['slug']}")
        print(f"      {' | '.join(info_parts)}")
        print()


def prompt_selection(articles: list[dict]) -> list[dict]:
    """Prompt user to select articles to publish as pods."""
    display_menu(articles)

    print("Enter article numbers (comma-separated), 'all', or 'q' to quit:")
    choice = input("> ").strip().lower()

    if choice in ("q", "quit", "exit", ""):
        return []
    if choice == "all":
        return articles

    try:
        indices = [int(x.strip()) - 1 for x in choice.split(",")]
        selected = [articles[i] for i in indices if 0 <= i < len(articles)]
        if not selected:
            print("No valid selections.")
            return []
        return selected
    except (ValueError, IndexError):
        print("Invalid input.")
        return []


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Publish pods to VIZflix courses")
    parser.add_argument("--course", help="Target course slug")
    parser.add_argument("--pod", help="Pod slug (article slug) to publish")
    args = parser.parse_args()

    articles = scan_articles()
    if not articles:
        print("No publishable articles found in output/articles/.")
        print("Each article needs a final.md file.")
        sys.exit(0)

    # Get available courses
    courses = get_available_courses()
    if not courses:
        print("No courses found. Create a course first with scripts/create_course.py")
        sys.exit(1)

    # If --pod specified, find that specific article
    if args.pod:
        selected = [a for a in articles if a["slug"] == args.pod]
        if not selected:
            print(f"  Error: Article '{args.pod}' not found in output/articles/")
            sys.exit(1)
    else:
        selected = prompt_selection(articles)
        if not selected:
            print("Nothing selected. Exiting.")
            sys.exit(0)

    # Select target course
    course = select_course(courses, preset=args.course)
    if not course:
        print("No course selected. Exiting.")
        sys.exit(0)

    course_slug = course["slug"]
    print(f"\n  Target course: {course['title']} ({course_slug})")

    # Query Drive for notebook IDs once
    print("\nQuerying Google Drive for notebook IDs...")
    drive_ids = get_drive_notebook_ids()
    if drive_ids:
        print(f"  Found {len(drive_ids)} notebooks on Drive")
    else:
        print("  No notebook IDs found (Colab URLs will be empty)")

    print(f"\nPublishing {len(selected)} pod(s) to course '{course_slug}'...")

    for article in selected:
        publish_pod(article, course_slug, drive_ids)

    print(f"\n{'=' * 64}")
    print(f"  Done! Published {len(selected)} pod(s) to '{course_slug}'.")
    print(f"  Run `npm run build` to verify.")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
