#!/usr/bin/env python3
"""Convert article markdown to Notion blocks and publish via API.

Usage:
  python publish_to_notion.py <article.md> --pod-json <pod.json> [--update]

When --update is given, the script finds the existing Notion page via
the notionUrl in pod.json, deletes all its child blocks, and re-adds
the new content (preserving the same page URL).
"""

import argparse
import json
import re
import sys
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

# Default figure URL mapping (used when no pod.json supplied)
FIGURE_URLS = {}


def parse_rich_text(text):
    """Convert markdown inline formatting to Notion rich text array."""
    parts = []
    i = 0
    while i < len(text):
        # Inline equation: $...$
        m = re.match(r'\$([^$]+)\$', text[i:])
        if m:
            parts.append({
                "type": "equation",
                "equation": {"expression": m.group(1)}
            })
            i += m.end()
            continue

        # Bold+italic: ***...*** or ___...___
        m = re.match(r'\*\*\*(.+?)\*\*\*', text[i:])
        if m:
            parts.append({
                "type": "text",
                "text": {"content": m.group(1)},
                "annotations": {"bold": True, "italic": True}
            })
            i += m.end()
            continue

        # Bold: **...**
        m = re.match(r'\*\*(.+?)\*\*', text[i:])
        if m:
            parts.append({
                "type": "text",
                "text": {"content": m.group(1)},
                "annotations": {"bold": True}
            })
            i += m.end()
            continue

        # Italic: *...*
        m = re.match(r'\*([^*]+?)\*', text[i:])
        if m:
            parts.append({
                "type": "text",
                "text": {"content": m.group(1)},
                "annotations": {"italic": True}
            })
            i += m.end()
            continue

        # Inline code: `...`
        m = re.match(r'`([^`]+)`', text[i:])
        if m:
            parts.append({
                "type": "text",
                "text": {"content": m.group(1)},
                "annotations": {"code": True}
            })
            i += m.end()
            continue

        # Plain text - accumulate until next special char
        end = i + 1
        while end < len(text):
            if text[end] in ('*', '`', '$'):
                break
            end += 1
        parts.append({
            "type": "text",
            "text": {"content": text[i:end]}
        })
        i = end

    return parts


def make_paragraph(text):
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {"rich_text": parse_rich_text(text)}
    }


def make_heading2(text):
    clean = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    return {
        "object": "block",
        "type": "heading_2",
        "heading_2": {"rich_text": [{"type": "text", "text": {"content": clean}}]}
    }


def make_heading3(text):
    clean = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    return {
        "object": "block",
        "type": "heading_3",
        "heading_3": {"rich_text": [{"type": "text", "text": {"content": clean}}]}
    }


def make_divider():
    return {"object": "block", "type": "divider", "divider": {}}


def make_image(url, caption=""):
    blocks = [
        {
            "object": "block",
            "type": "image",
            "image": {
                "type": "external",
                "external": {"url": url}
            }
        }
    ]
    if caption:
        blocks.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": caption}, "annotations": {"italic": True}}]
            }
        })
    return blocks


def make_equation(latex):
    return {
        "object": "block",
        "type": "equation",
        "equation": {"expression": latex}
    }


def make_code(code, language="python"):
    return {
        "object": "block",
        "type": "code",
        "code": {
            "rich_text": [{"type": "text", "text": {"content": code}}],
            "language": language
        }
    }


def make_bullet(text):
    return {
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {"rich_text": parse_rich_text(text)}
    }


def make_numbered(text):
    return {
        "object": "block",
        "type": "numbered_list_item",
        "numbered_list_item": {"rich_text": parse_rich_text(text)}
    }


def make_table(rows):
    width = max(len(r) for r in rows)
    table_rows = []
    for row in rows:
        cells = []
        for cell in row:
            cells.append(parse_rich_text(cell.strip()))
        while len(cells) < width:
            cells.append([{"type": "text", "text": {"content": ""}}])
        table_rows.append({
            "object": "block",
            "type": "table_row",
            "table_row": {"cells": cells}
        })
    return {
        "object": "block",
        "type": "table",
        "table": {
            "table_width": width,
            "has_column_header": True,
            "has_row_header": False,
            "children": table_rows
        }
    }


def parse_markdown(md_text, figure_urls):
    """Parse markdown text into Notion blocks."""
    blocks = []
    lines = md_text.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i]

        # Skip the H1 title (page title handles it)
        if line.startswith('# ') and not line.startswith('## '):
            i += 1
            continue

        # Empty line
        if not line.strip():
            i += 1
            continue

        # Divider
        if line.strip() == '---':
            blocks.append(make_divider())
            i += 1
            continue

        # Heading 3
        if line.startswith('### '):
            blocks.append(make_heading3(line[4:].strip()))
            i += 1
            continue

        # Heading 2
        if line.startswith('## '):
            blocks.append(make_heading2(line[3:].strip()))
            i += 1
            continue

        # Code block
        if line.strip().startswith('```'):
            lang_match = re.match(r'```(\w*)', line.strip())
            lang = lang_match.group(1) if lang_match else "plain text"
            if lang == "":
                lang = "plain text"
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing ```
            blocks.append(make_code('\n'.join(code_lines), lang))
            continue

        # Block equation: $$...$$
        if line.strip().startswith('$$'):
            if line.strip().endswith('$$') and len(line.strip()) > 4:
                latex = line.strip()[2:-2].strip()
                blocks.append(make_equation(latex))
                i += 1
                continue
            else:
                eq_lines = []
                i += 1
                while i < len(lines) and not lines[i].strip().startswith('$$'):
                    eq_lines.append(lines[i])
                    i += 1
                i += 1
                latex = '\n'.join(eq_lines).strip()
                blocks.append(make_equation(latex))
                continue

        # Image: ![alt](path)
        img_match = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', line.strip())
        if img_match:
            alt = img_match.group(1)
            path = img_match.group(2)
            url = figure_urls.get(path, path)
            caption = ""
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                cap_match = re.match(r'^\*(.+)\*$', lines[j].strip())
                if cap_match:
                    caption = cap_match.group(1)
                    i = j + 1
                else:
                    i += 1
            else:
                i += 1
            blocks.extend(make_image(url, caption))
            continue

        # Italic-only line (standalone caption)
        cap_match = re.match(r'^\*(.+)\*$', line.strip())
        if cap_match:
            blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": cap_match.group(1)}, "annotations": {"italic": True}}]
                }
            })
            i += 1
            continue

        # Table
        if '|' in line and i + 1 < len(lines) and re.match(r'^\|[-|: ]+\|$', lines[i + 1].strip()):
            rows = []
            header_cells = [c.strip() for c in line.strip().strip('|').split('|')]
            rows.append(header_cells)
            i += 2
            while i < len(lines) and '|' in lines[i] and lines[i].strip().startswith('|'):
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                rows.append(cells)
                i += 1
            blocks.append(make_table(rows))
            continue

        # Bullet list
        if re.match(r'^[-*] ', line.strip()):
            text = re.sub(r'^[-*] ', '', line.strip())
            blocks.append(make_bullet(text))
            i += 1
            continue

        # Numbered list
        num_match = re.match(r'^(\d+)\.\s+', line.strip())
        if num_match:
            text = re.sub(r'^\d+\.\s+', '', line.strip())
            blocks.append(make_numbered(text))
            i += 1
            continue

        # Regular paragraph
        para_lines = [line]
        i += 1
        while i < len(lines):
            next_line = lines[i]
            if (not next_line.strip() or
                next_line.startswith('#') or
                next_line.strip() == '---' or
                next_line.strip().startswith('```') or
                next_line.strip().startswith('$$') or
                re.match(r'!\[', next_line.strip()) or
                re.match(r'^\*[^*]+\*$', next_line.strip()) or
                re.match(r'^[-*] ', next_line.strip()) or
                re.match(r'^\d+\.\s+', next_line.strip()) or
                ('|' in next_line and i + 1 < len(lines) and
                 re.match(r'^\|[-|: ]+\|$', lines[i + 1].strip() if i + 1 < len(lines) else ''))):
                break
            para_lines.append(next_line)
            i += 1
        full_para = ' '.join(l.strip() for l in para_lines)
        if len(full_para) > 1900:
            sentences = re.split(r'(?<=[.!?]) ', full_para)
            chunk = ""
            for s in sentences:
                if len(chunk) + len(s) > 1900:
                    if chunk:
                        blocks.append(make_paragraph(chunk.strip()))
                    chunk = s
                else:
                    chunk += (" " if chunk else "") + s
            if chunk:
                blocks.append(make_paragraph(chunk.strip()))
        else:
            blocks.append(make_paragraph(full_para))

    return blocks


def create_page(title, blocks):
    """Create a new Notion page and add blocks in batches of 100."""
    first_batch = blocks[:100]
    remaining = blocks[100:]

    data = {
        "parent": {"page_id": PARENT_PAGE_ID},
        "icon": {"type": "emoji", "emoji": "\U0001f5a5\ufe0f"},
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

    append_remaining_blocks(page_id, remaining)
    return page_url


def update_page(page_id, blocks):
    """Update an existing Notion page: delete all child blocks, then add new ones."""
    print(f"Updating existing page: {page_id}")

    # Step 1: Get all existing child blocks
    all_block_ids = []
    url = f"https://api.notion.com/v1/blocks/{page_id}/children?page_size=100"
    while url:
        resp = requests.get(url, headers=HEADERS)
        if resp.status_code != 200:
            print(f"Error fetching children: {resp.status_code}")
            print(resp.text[:500])
            return None
        data = resp.json()
        for block in data.get("results", []):
            all_block_ids.append(block["id"])
        if data.get("has_more"):
            cursor = data.get("next_cursor")
            url = f"https://api.notion.com/v1/blocks/{page_id}/children?page_size=100&start_cursor={cursor}"
        else:
            url = None

    print(f"  Found {len(all_block_ids)} existing blocks to delete")

    # Step 2: Delete all existing blocks
    for bid in all_block_ids:
        resp = requests.delete(
            f"https://api.notion.com/v1/blocks/{bid}",
            headers=HEADERS
        )
        if resp.status_code != 200:
            print(f"  Warning: failed to delete block {bid}: {resp.status_code}")
        time.sleep(0.1)  # rate limiting

    print(f"  Deleted {len(all_block_ids)} blocks")

    # Step 3: Add new blocks in batches of 100
    remaining = blocks
    batch_num = 0
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

    # Get the page URL
    resp = requests.get(
        f"https://api.notion.com/v1/pages/{page_id}",
        headers=HEADERS
    )
    if resp.status_code == 200:
        return resp.json().get("url", f"https://notion.so/{page_id}")
    return f"https://notion.so/{page_id}"


def append_remaining_blocks(page_id, remaining):
    """Append remaining blocks in batches of 100."""
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


def extract_page_id_from_url(notion_url):
    """Extract Notion page ID from a Notion URL.

    URLs look like: https://www.notion.so/Page-Title-<32hex>
    The last 32 hex chars (no dashes) are the page ID.
    """
    # Match the 32-char hex at the end of the URL
    m = re.search(r'([0-9a-f]{32})$', notion_url.rstrip('/'))
    if m:
        raw = m.group(1)
        # Format as UUID: 8-4-4-4-12
        return f"{raw[:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:]}"
    return None


def build_figure_urls(pod_data):
    """Build figure URL mapping from pod.json figureUrls."""
    urls = {}
    figure_urls = pod_data.get("article", {}).get("figureUrls", {})
    for key, url in figure_urls.items():
        # Map both "figures/figure_1.png" and "figures/figure_1" formats
        urls[f"figures/{key}.png"] = url
        urls[f"figures/{key}"] = url
    return urls


def main():
    parser = argparse.ArgumentParser(description="Publish a markdown article to Notion")
    parser.add_argument("article", help="Path to the article markdown file")
    parser.add_argument("--pod-json", help="Path to pod.json for figure URLs and metadata")
    parser.add_argument("--update", action="store_true",
                        help="Update existing Notion page (uses notionUrl from pod.json)")
    args = parser.parse_args()

    # Read article
    with open(args.article) as f:
        md = f.read()

    # Load pod.json if provided
    pod_data = None
    figure_urls = {}
    if args.pod_json:
        with open(args.pod_json) as f:
            pod_data = json.load(f)
        figure_urls = build_figure_urls(pod_data)
        print(f"Loaded {len(figure_urls) // 2} figure URLs from pod.json")

    # Extract title from H1 in markdown
    title = "Untitled"
    for line in md.split('\n'):
        if line.startswith('# ') and not line.startswith('## '):
            title = line[2:].strip()
            # Remove any markdown formatting from title
            title = re.sub(r'[*_`]', '', title)
            break

    # If pod.json has a title, prefer that
    if pod_data and pod_data.get("title"):
        title = pod_data["title"]

    print(f"Title: {title}")
    print("Parsing markdown...")
    blocks = parse_markdown(md, figure_urls)
    print(f"Generated {len(blocks)} Notion blocks")

    if args.update and pod_data:
        notion_url = pod_data.get("article", {}).get("notionUrl", "")
        if notion_url:
            page_id = extract_page_id_from_url(notion_url)
            if page_id:
                url = update_page(page_id, blocks)
                if url:
                    print(f"\nUpdated: {url}")
                return
            else:
                print(f"Could not extract page ID from URL: {notion_url}")
                print("Creating new page instead...")
        else:
            print("No notionUrl found in pod.json, creating new page...")

    url = create_page(title, blocks)
    if url:
        print(f"\nPublished to: {url}")


if __name__ == "__main__":
    main()
