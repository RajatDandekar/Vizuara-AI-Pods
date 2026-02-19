#!/usr/bin/env python3
"""
Create a new course with pod placeholders for VIZflix.

Usage:
    python scripts/create_course.py                          # Interactive mode
    python scripts/create_course.py --slug nlp-basics \
        --title "NLP Basics" \
        --description "Learn NLP from scratch" \
        --difficulty intermediate \
        --pods "tokenization,word-embeddings,seq2seq-models"
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONTENT_DIR = PROJECT_ROOT / "content" / "courses"
CATALOG_PATH = CONTENT_DIR / "catalog.json"


def slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    import re
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug


def load_catalog() -> dict:
    if CATALOG_PATH.exists():
        with open(CATALOG_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"courses": []}


def save_catalog(catalog: dict):
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
        f.write("\n")


def course_exists(slug: str) -> bool:
    catalog = load_catalog()
    return any(c["slug"] == slug for c in catalog["courses"])


def prompt_pods() -> list[dict]:
    """Interactively collect pod definitions."""
    pods = []
    print()
    print("  Add pods (press Enter with empty title to finish):")
    print()
    order = 1
    while True:
        title = input(f"  Pod {order} title: ").strip()
        if not title:
            break

        default_slug = slugify(title)
        slug = input(f"    Slug [{default_slug}]: ").strip() or default_slug

        desc = input(f"    Description (Enter to auto-generate): ").strip()
        if not desc:
            desc = f"Coming soon: {title}"

        pods.append({
            "title": title,
            "slug": slug,
            "description": desc,
            "order": order,
        })
        order += 1
        print()

    return pods


def create_course(slug: str, title: str, description: str,
                  difficulty: str, tags: list[str],
                  pods: list[dict], status: str = "draft"):
    """Create course.json, pod.json placeholders, and catalog entry."""

    if course_exists(slug):
        print(f"  Error: Course '{slug}' already exists in catalog.")
        sys.exit(1)

    course_dir = CONTENT_DIR / slug
    course_dir.mkdir(parents=True, exist_ok=True)

    # Build pod entries for course.json
    pod_entries = []
    for pod in pods:
        pod_entries.append({
            "slug": pod["slug"],
            "title": pod["title"],
            "description": pod["description"],
            "order": pod["order"],
            "notebookCount": 0,
            "estimatedHours": 1,
            "hasCaseStudy": False,
        })

        # Create pod directory with placeholder pod.json
        pod_dir = course_dir / "pods" / pod["slug"]
        pod_dir.mkdir(parents=True, exist_ok=True)

        pod_manifest = {
            "title": pod["title"],
            "slug": pod["slug"],
            "description": pod["description"],
            "difficulty": difficulty,
            "estimatedHours": 1,
            "prerequisites": [],
            "tags": [],
            "article": {"notionUrl": None, "figureUrls": {}},
            "notebooks": [],
            "courseSlug": slug,
            "order": pod["order"],
        }

        with open(pod_dir / "pod.json", "w", encoding="utf-8") as f:
            json.dump(pod_manifest, f, indent=2, ensure_ascii=False)
            f.write("\n")

    # Build course.json
    course_manifest = {
        "title": title,
        "slug": slug,
        "description": description,
        "difficulty": difficulty,
        "estimatedHours": len(pods),
        "tags": tags,
        "pods": pod_entries,
    }

    with open(course_dir / "course.json", "w", encoding="utf-8") as f:
        json.dump(course_manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # Update catalog
    catalog = load_catalog()
    catalog["courses"].append({
        "slug": slug,
        "title": title,
        "description": description,
        "difficulty": difficulty,
        "estimatedHours": len(pods),
        "tags": tags,
        "podCount": len(pods),
        "totalNotebooks": 0,
        "status": status,
    })
    catalog["courses"].sort(key=lambda c: c["title"])
    save_catalog(catalog)

    # Summary
    print()
    print(f"  {'=' * 50}")
    print(f"  Course created: {title}")
    print(f"  {'=' * 50}")
    print(f"  Slug:       {slug}")
    print(f"  Difficulty:  {difficulty}")
    print(f"  Status:      {status}")
    print(f"  Pods:        {len(pods)}")
    print()
    for pod in pods:
        print(f"    {pod['order']}. {pod['title']} ({pod['slug']})")
    print()
    print(f"  Files created:")
    print(f"    content/courses/{slug}/course.json")
    for pod in pods:
        print(f"    content/courses/{slug}/pods/{pod['slug']}/pod.json")
    print(f"    catalog.json updated")
    print()
    print(f"  Next steps:")
    print(f"    1. For each pod, run the content skills:")
    print(f"       /write-substack, /generate-notebooks, /generate-case-study")
    print(f"    2. Publish each pod:")
    print(f"       python scripts/publish_course.py --course {slug} --pod <podSlug>")
    print(f"    3. When ready to go live:")
    print(f"       python scripts/manage_courses.py show {slug}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Create a new VIZflix course")
    parser.add_argument("--slug", help="Course slug")
    parser.add_argument("--title", help="Course title")
    parser.add_argument("--description", help="Course description")
    parser.add_argument("--difficulty", choices=["beginner", "intermediate", "advanced"],
                        default="intermediate")
    parser.add_argument("--tags", help="Comma-separated tags")
    parser.add_argument("--pods", help="Comma-separated pod slugs (titles auto-generated from slugs)")
    parser.add_argument("--status", choices=["draft", "live", "upcoming"], default="draft")
    args = parser.parse_args()

    print()
    print("=" * 54)
    print("   VIZUARA COURSE CREATOR")
    print("=" * 54)

    # Course details
    title = args.title or input("\n  Course title: ").strip()
    if not title:
        print("  Title is required.")
        sys.exit(1)

    default_slug = args.slug or slugify(title)
    slug = default_slug if args.slug else (input(f"  Slug [{default_slug}]: ").strip() or default_slug)

    description = args.description or input("  Description: ").strip()
    if not description:
        description = title

    if not args.difficulty:
        diff = input("  Difficulty [beginner/intermediate/advanced] (default: intermediate): ").strip()
        difficulty = diff if diff in ("beginner", "intermediate", "advanced") else "intermediate"
    else:
        difficulty = args.difficulty

    if args.tags:
        tags = [t.strip() for t in args.tags.split(",")]
    else:
        tags_input = input("  Tags (comma-separated): ").strip()
        tags = [t.strip() for t in tags_input.split(",")] if tags_input else []

    # Pods
    if args.pods:
        pod_slugs = [s.strip() for s in args.pods.split(",")]
        pods = []
        for i, ps in enumerate(pod_slugs, 1):
            # Convert slug to title: "word-embeddings" → "Word Embeddings"
            pod_title = ps.replace("-", " ").title()
            pods.append({
                "title": pod_title,
                "slug": ps,
                "description": f"Coming soon: {pod_title}",
                "order": i,
            })
    else:
        pods = prompt_pods()

    if not pods:
        print("  At least one pod is required.")
        sys.exit(1)

    create_course(slug, title, description, difficulty, tags, pods, args.status)


if __name__ == "__main__":
    main()
