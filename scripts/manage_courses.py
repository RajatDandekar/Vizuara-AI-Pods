#!/usr/bin/env python3
"""
Manage course visibility on the VIZflix dashboard.

Usage:
    python scripts/manage_courses.py                  # List all courses with status
    python scripts/manage_courses.py show 3           # Set course #3 to live
    python scripts/manage_courses.py hide 3           # Set course #3 to draft (hidden)
    python scripts/manage_courses.py show 1,3,5       # Show multiple courses
    python scripts/manage_courses.py hide all         # Hide everything
    python scripts/manage_courses.py show all         # Show everything
    python scripts/manage_courses.py only 3           # Show ONLY course #3, hide rest
    python scripts/manage_courses.py only 1,3,5       # Show ONLY these, hide rest

You can also use slugs instead of numbers:
    python scripts/manage_courses.py show diffusion-llms
    python scripts/manage_courses.py only world-action-models,diffusion-llms
"""

import json
import sys
from pathlib import Path

CATALOG_PATH = Path(__file__).resolve().parent.parent / "content" / "courses" / "catalog.json"


def load_catalog() -> dict:
    with open(CATALOG_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_catalog(catalog: dict):
    with open(CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
        f.write("\n")


def list_courses(catalog: dict):
    courses = catalog["courses"]
    print()
    print(f"  {'#':>3}  {'Status':>9}  {'Pods':>5}  {'NBs':>4}  Title")
    print(f"  {'─' * 3}  {'─' * 9}  {'─' * 5}  {'─' * 4}  {'─' * 45}")
    for i, c in enumerate(courses, 1):
        status = c.get("status", "live")
        marker = {
            "live": "\033[32m    live\033[0m",      # green
            "upcoming": "\033[33mupcoming\033[0m",   # yellow
            "draft": "\033[90m   draft\033[0m",      # gray
            "archived": "\033[90marchived\033[0m",   # gray
        }.get(status, status)
        pods = c.get("podCount", 0)
        nbs = c.get("totalNotebooks", c.get("notebookCount", 0))
        print(f"  {i:3d}  {marker:>18s}  {pods:5d}  {nbs:4d}  {c['title'][:45]}")
    print()
    live_count = sum(1 for c in courses if c.get("status", "live") == "live")
    total_pods = sum(c.get("podCount", 0) for c in courses)
    print(f"  {live_count} live / {len(courses)} total courses, {total_pods} total pods")
    print()


def resolve_targets(catalog: dict, spec: str) -> list[int]:
    """Resolve a target spec like '3', '1,3,5', 'all', or 'slug-name' to 0-based indices."""
    courses = catalog["courses"]

    if spec.strip().lower() == "all":
        return list(range(len(courses)))

    indices = []
    for part in spec.split(","):
        part = part.strip()
        # Try as number first
        try:
            num = int(part)
            if 1 <= num <= len(courses):
                indices.append(num - 1)
            else:
                print(f"  Error: #{num} is out of range (1-{len(courses)})")
                sys.exit(1)
            continue
        except ValueError:
            pass

        # Try as slug
        found = False
        for i, c in enumerate(courses):
            if c["slug"] == part:
                indices.append(i)
                found = True
                break
        if not found:
            print(f"  Error: '{part}' is not a valid number or slug")
            sys.exit(1)

    return indices


def ensure_thumbnail(course: dict):
    """Auto-add thumbnail when a course goes live.

    RULE: Thumbnails must use PaperBanana figures only (polished, publication-grade).
    Never use Excalidraw figures (hand-drawn whiteboard style) as thumbnails.
    Default: /courses/{slug}/figures/figure_1.png — verify it's PaperBanana before publishing.
    """
    if course.get("thumbnail"):
        return
    slug = course["slug"]
    thumb = f"/courses/{slug}/figures/figure_1.png"
    course["thumbnail"] = thumb
    print(f"  + thumbnail added: {thumb}")
    print(f"    \033[33m⚠ Verify this is a PaperBanana figure (not Excalidraw)!\033[0m")


def set_status(catalog: dict, indices: list[int], status: str):
    for i in indices:
        old = catalog["courses"][i].get("status", "live")
        if old == status:
            continue
        catalog["courses"][i]["status"] = status
        title = catalog["courses"][i]["title"][:45]
        print(f"  {old:>8s} -> {status:<8s}  {title}")
        if status == "live":
            ensure_thumbnail(catalog["courses"][i])


def main():
    catalog = load_catalog()

    if len(sys.argv) < 2:
        list_courses(catalog)
        print("  Commands: show <N>, hide <N>, only <N>")
        print("  Examples: show 3    hide 1,5    only 12    show all")
        print()
        return

    cmd = sys.argv[1].lower()
    if cmd == "list":
        list_courses(catalog)
        return

    if cmd not in ("show", "hide", "only"):
        print(f"  Unknown command: {cmd}")
        print(f"  Valid commands: show, hide, only, list")
        sys.exit(1)

    if len(sys.argv) < 3:
        print(f"  Usage: manage_courses.py {cmd} <number|slug|all>")
        sys.exit(1)

    spec = sys.argv[2]
    targets = resolve_targets(catalog, spec)

    if cmd == "show":
        set_status(catalog, targets, "live")
    elif cmd == "hide":
        set_status(catalog, targets, "draft")
    elif cmd == "only":
        # Hide everything first, then show only targets
        all_indices = list(range(len(catalog["courses"])))
        # Don't touch "upcoming" courses — only toggle between live/draft
        for i in all_indices:
            if catalog["courses"][i].get("status") == "upcoming":
                continue
            if i in targets:
                set_status(catalog, [i], "live")
            else:
                set_status(catalog, [i], "draft")

    save_catalog(catalog)
    print()
    live_count = sum(1 for c in catalog["courses"] if c.get("status", "live") == "live")
    print(f"  Done. {live_count} live courses.")
    print(f"  Run: npm run build && npm run dev")
    print()


if __name__ == "__main__":
    main()
