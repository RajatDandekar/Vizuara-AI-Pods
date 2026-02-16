#!/usr/bin/env python3
"""
Interactive course publisher for VIZflix.

Scans output/articles/ for completed articles, detects associated notebooks
and narration, and lets you choose which articles to publish to the dashboard.

Usage:
    python scripts/publish_course.py
"""

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

        # Narration — check per-slug first, then flat (legacy), then notebooks
        narr_dir = NARRATION_DIR / slug
        has_narration = narr_dir.exists() and any(narr_dir.glob("*.mp3"))
        if not has_narration:
            # Check if any notebook already has injected narration cells
            for nb_path in notebooks:
                if _detect_notebook_narration(nb_path):
                    has_narration = True
                    break

        # Case study
        case_study = detect_case_study(slug)

        already_published = (CONTENT_DIR / slug / "course.json").exists()

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


# ── Case study detection ─────────────────────────────────────────────────────


def detect_case_study(slug: str) -> dict | None:
    """Detect a case study in output/case-studies/{slug}/ and extract metadata."""
    cs_dir = CASE_STUDIES_DIR / slug
    cs_md = cs_dir / "case_study.md"
    if not cs_md.exists():
        return None

    text = cs_md.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Extract title from first # heading (e.g., "# Case Study: Autonomous Pick...")
    title = "Case Study"
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            raw = stripped[2:].strip()
            # Remove "Case Study: " prefix if present
            title = re.sub(r"^Case Study:\s*", "", raw).strip()
            break

    # Extract subtitle from first ## heading
    subtitle = ""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## ") and not stripped.startswith("### "):
            subtitle = stripped[3:].strip()
            break

    # Extract company name from "### Company Profile: ..." heading
    company = ""
    for line in lines:
        stripped = line.strip()
        match = re.match(r"^###\s*Company Profile:\s*(.+)", stripped)
        if match:
            company = match.group(1).strip()
            break

    # Extract industry from "### Industry: ..." heading
    industry = ""
    for line in lines:
        stripped = line.strip()
        match = re.match(r"^###\s*Industry:\s*(.+)", stripped)
        if match:
            industry = match.group(1).strip()
            break

    # Extract description — first paragraph after "### Business Challenge"
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
    """Check if a notebook has embedded narration cells (injected by inject_narration.py)."""
    try:
        with open(notebook_path, encoding="utf-8") as f:
            nb = json.load(f)
        for cell in nb.get("cells", []):
            # Check for narration tag in metadata
            tags = cell.get("metadata", {}).get("tags", [])
            if "narration" in tags:
                return True
            # Check for _DRIVE_ID + audio segment pattern (narration player cells)
            source = "".join(cell.get("source", []))
            if "_DRIVE_ID" in source and "_SEG" in source:
                return True
            # Also check for the gdown narration download pattern
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
    """Re-upload notebooks to Drive so Colab gets the latest version (with narration)."""
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


def build_notebook_meta(slug: str, notebooks: list[Path], drive_ids: dict) -> list[dict]:
    """Build NotebookMeta entries for course.json."""
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
                    # Remove emoji prefixes
                    title = re.sub(r"^[^\w]+", "", title).strip()
                    break
            if title != f"Notebook {i}":
                # Found title — now look for objective in next lines
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

        # Estimate minutes (non-narration cells * 1.5)
        non_narration = [c for c in cells
                         if "narration" not in c.get("metadata", {}).get("tags", [])]
        estimated_minutes = max(30, round(len(non_narration) * 1.5))

        # Colab URL from Drive ID
        drive_id = drive_ids.get(nb_path.name, "")
        colab_url = f"https://colab.research.google.com/drive/{drive_id}" if drive_id else ""

        # Narration: just flag whether the notebook has narration cells
        # (audio is served via Google Drive/gdown in Colab, not from public/)
        has_narration = _detect_notebook_narration(nb_path)

        # Derive a URL-friendly slug from filename
        stem = nb_path.stem  # e.g., "01_world_models_first_principles"
        parts = stem.split("_", 1)
        clean_slug = f"{parts[0]}-{parts[1]}" if len(parts) > 1 else stem

        metas.append({
            "title": title,
            "slug": clean_slug,
            "objective": objective,
            "colabUrl": colab_url,
            "downloadPath": f"/notebooks/{slug}/{nb_path.name}",
            "hasNarration": has_narration,
            "estimatedMinutes": estimated_minutes,
            "todoCount": todo_count,
            "order": i,
        })

    return metas


# ── Publishing ────────────────────────────────────────────────────────────────


def publish_article(article: dict, drive_ids: dict):
    """Publish a single article to the content directory."""
    slug = article["slug"]
    print(f"\n{'─' * 60}")
    print(f"  Publishing: {article['title']}")
    print(f"{'─' * 60}")

    # 1. Create content directory
    course_dir = CONTENT_DIR / slug
    course_dir.mkdir(parents=True, exist_ok=True)

    # 2. Copy final.md → article.md
    shutil.copy2(article["dir"] / "final.md", course_dir / "article.md")
    print(f"  [+] Copied article.md")

    # 3. Copy figures to public/
    if article["figures"]:
        public_fig_dir = PUBLIC_DIR / slug / "figures"
        public_fig_dir.mkdir(parents=True, exist_ok=True)
        for fig in article["figures"]:
            shutil.copy2(fig, public_fig_dir / fig.name)
        print(f"  [+] Copied {len(article['figures'])} figures")

    # 4. Copy equations to public/ (if any)
    if article["equations"]:
        public_eq_dir = PUBLIC_DIR / slug / "equations"
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

        notebook_metas = build_notebook_meta(slug, article["notebooks"], drive_ids)

        # Copy notebook files to public/
        public_nb_dir = PUBLIC_NB_DIR / slug
        public_nb_dir.mkdir(parents=True, exist_ok=True)
        for nb in article["notebooks"]:
            shutil.copy2(nb, public_nb_dir / nb.name)
        print(f"  [+] Copied {len(article['notebooks'])} notebooks to public/")

        # Re-upload notebooks to Drive so Colab gets the latest version
        # (including any narration cells injected into the notebooks)
        print(f"  [~] Syncing notebooks to Google Drive...")
        _sync_notebooks_to_drive(article["notebooks"])

        missing = [m for m in notebook_metas if not m["colabUrl"]]
        if missing:
            print(f"  [!] Warning: {len(missing)} notebooks missing Drive IDs (no Colab URL)")
            print(f"      Upload notebooks first: python scripts/upload_notebooks_to_drive.py --slug {slug}")
    else:
        print(f"  [i] No notebooks found — publishing as article-only course")

    # 6. Narration — audio lives only in output/narration/{slug}/
    # Colab notebooks download segments from Google Drive (via gdown),
    # so no copy to public/ is needed.
    if article["has_narration"]:
        print(f"  [i] Narration detected (source: output/narration/{slug}/)")

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

        # Copy files to public/
        public_cs_dir = PUBLIC_CS_DIR / slug
        public_cs_dir.mkdir(parents=True, exist_ok=True)

        # Copy case_study.md to public/ (for static download) and content/ (for server-side rendering)
        shutil.copy2(cs["dir"] / "case_study.md", public_cs_dir / "case_study.md")
        shutil.copy2(cs["dir"] / "case_study.md", course_dir / "case_study.md")

        # Copy PDF if available
        if cs["has_pdf"]:
            shutil.copy2(cs["dir"] / "case_study.pdf", public_cs_dir / "case_study.pdf")

        # Copy notebook if available
        if cs["has_notebook"]:
            shutil.copy2(cs["dir"] / "case_study_notebook.ipynb", public_cs_dir / "case_study_notebook.ipynb")

        # Upload case study notebook to Drive with slug-prefixed name
        # (each notebook needs a unique name since they share one flat Drive folder)
        cs_colab_url = ""
        if cs["has_notebook"]:
            cs_drive_name = f"{slug}-case_study_notebook.ipynb"
            cs_src = cs["dir"] / "case_study_notebook.ipynb"
            _upload_case_study_notebook(cs_src, cs_drive_name)
            # Look up Drive ID by the slug-prefixed name
            cs_drive_id = drive_ids.get(cs_drive_name, "")
            if not cs_drive_id:
                # Re-query Drive to pick up the just-uploaded file
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
            "pdfPath": f"/case-studies/{slug}/case_study.pdf" if cs["has_pdf"] else "",
            "colabUrl": cs_colab_url,
            "notebookPath": f"/case-studies/{slug}/case_study_notebook.ipynb" if cs["has_notebook"] else "",
        }
        print(f"  [+] Published case study files")
    else:
        print(f"\n  [i] No case study found for this course")

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

    # 8. Build course.json
    course_manifest = {
        "title": article["title"],
        "slug": slug,
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
    }

    if case_study_meta:
        course_manifest["caseStudy"] = case_study_meta

    with open(course_dir / "course.json", "w", encoding="utf-8") as f:
        json.dump(course_manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"  [+] Generated course.json")

    # 9. Update catalog
    update_catalog(slug, article["title"], article["description"],
                   difficulty, est_hours, tags, len(notebook_metas))
    print(f"  [+] Updated catalog.json")


def update_catalog(slug: str, title: str, description: str,
                   difficulty: str, est_hours: int, tags: list[str],
                   nb_count: int):
    """Add or update a course entry in catalog.json (idempotent)."""
    if CATALOG_PATH.exists():
        with open(CATALOG_PATH, encoding="utf-8") as f:
            catalog = json.load(f)
    else:
        CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        catalog = {"courses": []}

    # Remove existing entry for this slug
    catalog["courses"] = [c for c in catalog["courses"] if c["slug"] != slug]

    # Add new entry
    catalog["courses"].append({
        "slug": slug,
        "title": title,
        "description": description,
        "difficulty": difficulty,
        "estimatedHours": est_hours,
        "tags": tags,
        "notebookCount": nb_count,
        "status": "live",
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
    print("   VIZUARA COURSE PUBLISHER")
    print("=" * 64)
    print()

    for i, art in enumerate(articles, 1):
        status = "PUBLISHED" if art["already_published"] else "NEW"
        nb_count = len(art["notebooks"])

        # Build info line
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
    """Prompt user to select articles to publish."""
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
    articles = scan_articles()
    if not articles:
        print("No publishable articles found in output/articles/.")
        print("Each article needs a final.md file.")
        sys.exit(0)

    selected = prompt_selection(articles)
    if not selected:
        print("Nothing selected. Exiting.")
        sys.exit(0)

    # Query Drive for notebook IDs once (shared across all articles)
    print("\nQuerying Google Drive for notebook IDs...")
    drive_ids = get_drive_notebook_ids()
    if drive_ids:
        print(f"  Found {len(drive_ids)} notebooks on Drive")
    else:
        print("  No notebook IDs found (Colab URLs will be empty)")

    print(f"\nPublishing {len(selected)} course(s)...")

    for article in selected:
        publish_article(article, drive_ids)

    print(f"\n{'=' * 64}")
    print(f"  Done! Published {len(selected)} course(s).")
    print(f"  Run `npm run build` to verify.")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
