#!/usr/bin/env python3
"""Publish a Vizuara article to Notion.

Reusable tool — takes article path, title, and figure URLs as arguments.

Usage:
    python publish_to_notion.py \
      --article-dir output/articles/my-topic/ \
      --title "My Article Title" \
      --figure-urls figure_urls.json

    The figure_urls.json file should map local paths to lh3 URLs:
    {
      "figures/figure_1.png": "https://lh3.googleusercontent.com/d/FILE_ID=w2000",
      ...
    }
"""

import argparse
import json
import re
import time

import requests

NOTION_TOKEN = "ntn_Sw5325638252XON3aR1OpyG2D5ENwO23csbWrO2506m0AT"
PARENT_PAGE_ID = "30501fbe-4771-80d8-93bb-f1079b9d432a"
NOTION_VERSION = "2022-06-28"

HEADERS = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Notion-Version": NOTION_VERSION,
    "Content-Type": "application/json",
}

FIGURE_URLS = {}  # Populated from --figure-urls arg


def parse_rich_text(text):
    """Parse markdown inline formatting to Notion rich text array."""
    parts = []
    i = 0
    while i < len(text):
        # Inline code `...`
        m = re.match(r'`([^`]+)`', text[i:])
        if m:
            parts.append({"type": "text", "text": {"content": m.group(1)}, "annotations": {"code": True}})
            i += m.end()
            continue

        # Inline math $...$
        m = re.match(r'\$([^$]+)\$', text[i:])
        if m:
            parts.append({"type": "equation", "equation": {"expression": m.group(1)}})
            i += m.end()
            continue

        # Bold **...**
        m = re.match(r'\*\*(.+?)\*\*', text[i:])
        if m:
            parts.append({"type": "text", "text": {"content": m.group(1)}, "annotations": {"bold": True}})
            i += m.end()
            continue

        # Italic *...*
        m = re.match(r'\*([^*]+?)\*', text[i:])
        if m:
            parts.append({"type": "text", "text": {"content": m.group(1)}, "annotations": {"italic": True}})
            i += m.end()
            continue

        # Link [text](url)
        m = re.match(r'\[([^\]]+)\]\(([^)]+)\)', text[i:])
        if m:
            parts.append({"type": "text", "text": {"content": m.group(1), "link": {"url": m.group(2)}}})
            i += m.end()
            continue

        # Plain text - collect until next special char
        end = i + 1
        while end < len(text) and text[end] not in ('`', '$', '*', '['):
            end += 1
        parts.append({"type": "text", "text": {"content": text[i:end]}})
        i = end

    return parts if parts else [{"type": "text", "text": {"content": text}}]


def md_to_notion_blocks(md_content):
    """Convert markdown to Notion block objects."""
    blocks = []
    lines = md_content.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i]

        # Skip empty lines
        if not line.strip():
            i += 1
            continue

        # Horizontal rule
        if line.strip() == '---':
            blocks.append({"object": "block", "type": "divider", "divider": {}})
            i += 1
            continue

        # Heading 1 (#)
        m = re.match(r'^# (.+)$', line)
        if m:
            blocks.append({
                "object": "block", "type": "heading_1",
                "heading_1": {"rich_text": parse_rich_text(m.group(1))}
            })
            i += 1
            continue

        # Heading 2 (##)
        m = re.match(r'^## (.+)$', line)
        if m:
            blocks.append({
                "object": "block", "type": "heading_2",
                "heading_2": {"rich_text": parse_rich_text(m.group(1))}
            })
            i += 1
            continue

        # Heading 3 (###)
        m = re.match(r'^### (.+)$', line)
        if m:
            blocks.append({
                "object": "block", "type": "heading_3",
                "heading_3": {"rich_text": parse_rich_text(m.group(1))}
            })
            i += 1
            continue

        # Display equation block $$...$$ (multi-line)
        if line.strip() == '$$':
            eq_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() != '$$':
                eq_lines.append(lines[i])
                i += 1
            i += 1  # skip closing $$
            expression = '\n'.join(eq_lines).strip()
            blocks.append({
                "object": "block", "type": "equation",
                "equation": {"expression": expression}
            })
            continue

        # Inline display equation $$...$$ on one line
        m = re.match(r'^\$\$(.+)\$\$$', line.strip())
        if m:
            blocks.append({
                "object": "block", "type": "equation",
                "equation": {"expression": m.group(1).strip()}
            })
            i += 1
            continue

        # Code block
        if line.strip().startswith('```'):
            lang_match = re.match(r'^```(\w*)', line.strip())
            lang = lang_match.group(1) if lang_match else "plain text"
            if not lang:
                lang = "plain text"
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing ```
            code_text = '\n'.join(code_lines)
            blocks.append({
                "object": "block", "type": "code",
                "code": {
                    "rich_text": [{"type": "text", "text": {"content": code_text}}],
                    "language": lang
                }
            })
            continue

        # Image ![alt](url) followed by italic caption
        m = re.match(r'^!\[([^\]]*)\]\(([^)]+)\)', line)
        if m:
            img_path = m.group(2)
            img_url = FIGURE_URLS.get(img_path, img_path)

            blocks.append({
                "object": "block", "type": "image",
                "image": {
                    "type": "external",
                    "external": {"url": img_url}
                }
            })

            # Check next non-empty line for italic caption
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].strip().startswith('*') and lines[j].strip().endswith('*'):
                caption_text = lines[j].strip()[1:-1]
                blocks.append({
                    "object": "block", "type": "paragraph",
                    "paragraph": {"rich_text": [{"type": "text", "text": {"content": caption_text}, "annotations": {"italic": True}}]}
                })
                i = j + 1
            else:
                i += 1
            continue

        # Standalone italic line (caption not preceded by image)
        cap_match = re.match(r'^\*([^*]+)\*$', line.strip())
        if cap_match:
            blocks.append({
                "object": "block", "type": "paragraph",
                "paragraph": {"rich_text": [{"type": "text", "text": {"content": cap_match.group(1)}, "annotations": {"italic": True}}]}
            })
            i += 1
            continue

        # Blockquote
        m = re.match(r'^> (.+)$', line)
        if m:
            blocks.append({
                "object": "block", "type": "quote",
                "quote": {"rich_text": parse_rich_text(m.group(1))}
            })
            i += 1
            continue

        # Numbered list
        m = re.match(r'^(\d+)\.\s+(.+)$', line)
        if m:
            blocks.append({
                "object": "block", "type": "numbered_list_item",
                "numbered_list_item": {"rich_text": parse_rich_text(m.group(2))}
            })
            i += 1
            continue

        # Bulleted list
        m = re.match(r'^[-*] (.+)$', line)
        if m:
            blocks.append({
                "object": "block", "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": parse_rich_text(m.group(1))}
            })
            i += 1
            continue

        # Table (simple | delimited)
        if '|' in line and i + 1 < len(lines) and '---' in lines[i + 1]:
            headers = [c.strip() for c in line.split('|') if c.strip()]
            i += 2  # Skip header and separator
            rows = [headers]
            while i < len(lines) and '|' in lines[i] and lines[i].strip().startswith('|'):
                row = [c.strip() for c in lines[i].split('|') if c.strip()]
                rows.append(row)
                i += 1

            table_width = len(headers)
            table_rows = []
            for row in rows:
                cells = []
                for j in range(table_width):
                    cell_text = row[j] if j < len(row) else ""
                    cells.append(parse_rich_text(cell_text))
                table_rows.append({"type": "table_row", "table_row": {"cells": cells}})

            blocks.append({
                "object": "block", "type": "table",
                "table": {
                    "table_width": table_width,
                    "has_column_header": True,
                    "has_row_header": False,
                    "children": table_rows
                }
            })
            continue

        # Regular paragraph
        if line.strip():
            blocks.append({
                "object": "block", "type": "paragraph",
                "paragraph": {"rich_text": parse_rich_text(line.strip())}
            })
        i += 1

    return blocks


def create_page(title, blocks):
    """Create a Notion page and add blocks in batches of 100."""
    first_batch = blocks[:100]
    remaining = blocks[100:]

    data = {
        "parent": {"page_id": PARENT_PAGE_ID},
        "icon": {"type": "emoji", "emoji": "📝"},
        "properties": {
            "title": [{"text": {"content": title}}]
        },
        "children": first_batch
    }

    resp = requests.post(
        "https://api.notion.com/v1/pages",
        headers=HEADERS,
        json=data
    )

    if resp.status_code != 200:
        print(f"Error creating page: {resp.status_code}")
        print(resp.text[:500])
        return None

    page = resp.json()
    page_id = page["id"]
    page_url = page["url"]
    print(f"Page created: {page_url}")
    print(f"  First batch: {len(first_batch)} blocks")

    # Append remaining blocks in batches of 100
    batch_num = 1
    while remaining:
        batch = remaining[:100]
        remaining = remaining[100:]
        batch_num += 1
        time.sleep(0.4)

        resp = requests.patch(
            f"https://api.notion.com/v1/blocks/{page_id}/children",
            headers=HEADERS,
            json={"children": batch}
        )

        if resp.status_code != 200:
            print(f"  Error appending batch {batch_num}: {resp.status_code}")
            print(f"  {resp.text[:300]}")
        else:
            print(f"  Batch {batch_num} appended ({len(batch)} blocks)")

    return page_url


def main():
    parser = argparse.ArgumentParser(description="Publish a Vizuara article to Notion")
    parser.add_argument("--article-dir", required=True, help="Path to article directory (contains final.md)")
    parser.add_argument("--title", required=True, help="Article title for Notion page")
    parser.add_argument("--figure-urls", required=True, help="JSON file mapping figure paths to lh3 URLs")
    parser.add_argument("--icon", default="📝", help="Emoji icon for the page")
    args = parser.parse_args()

    # Load figure URLs
    global FIGURE_URLS
    with open(args.figure_urls) as f:
        FIGURE_URLS = json.load(f)
    print(f"Loaded {len(FIGURE_URLS)} figure URLs")

    # Read article
    article_path = f"{args.article_dir}/final.md"
    with open(article_path) as f:
        md = f.read()

    # Skip the H1 title line since it becomes the page title
    lines = md.split('\n')
    start = 0
    for idx, line in enumerate(lines):
        if line.startswith('# '):
            start = idx + 1
            break
    md_body = '\n'.join(lines[start:])

    print("Parsing markdown...")
    blocks = md_to_notion_blocks(md_body)
    print(f"Generated {len(blocks)} Notion blocks")

    url = create_page(args.title, blocks)
    if url:
        print(f"\nPublished to: {url}")


if __name__ == "__main__":
    main()
