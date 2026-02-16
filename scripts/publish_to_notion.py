#!/usr/bin/env python3
"""Convert final.md to Notion blocks and publish via API."""

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

# Figure URL mapping — use lh3.googleusercontent.com direct URLs (Notion-compatible)
# Uniquely named as 5d-parallelism-figure_N.png to avoid Drive filename collisions
FIGURE_URLS = {
    "figures/figure_1.png": "https://lh3.googleusercontent.com/d/1_IY1ySUrWNzcDWgh8kOitbwj3s_lyzFS=w2000",
    "figures/figure_2.png": "https://lh3.googleusercontent.com/d/1em_WMJfeWz4eBWazRq5jseF1dbYU7SO5=w2000",
    "figures/figure_3.png": "https://lh3.googleusercontent.com/d/1W5w__oHbApNjW_27ijPSRiGkXNFTMogM=w2000",
    "figures/figure_4.png": "https://lh3.googleusercontent.com/d/15qugrILaZIkZvyyInGktoRaeeBhN0tM_=w2000",
    "figures/figure_5.png": "https://lh3.googleusercontent.com/d/1TpsUvrPtVk26_iy7wVsBhBF-q8P3M35k=w2000",
    "figures/figure_6.png": "https://lh3.googleusercontent.com/d/1RRR_OOkE-8rVG9Fmwb75Q77fy7tG1z-7=w2000",
    "figures/figure_7.png": "https://lh3.googleusercontent.com/d/1rW9yMt5L7DX4XH4JUZ1IfGjgQLcE8wBj=w2000",
    "figures/figure_8.png": "https://lh3.googleusercontent.com/d/1DrrKxQcc98g2WpKMEMbb-wamEpZCPJ1U=w2000",
    "figures/figure_9.png": "https://lh3.googleusercontent.com/d/1g3r8O4qrF8L4e_x9m8ifd-ZW6P6peKCp=w2000",
    "figures/figure_10.png": "https://lh3.googleusercontent.com/d/1FuD6nY1HIM3MMuF69Yd0NBpeIkV59Soj=w2000",
}


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
    """Create a paragraph block with rich text."""
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {"rich_text": parse_rich_text(text)}
    }


def make_heading2(text):
    """Create a heading_2 block."""
    # Strip markdown bold from headings
    clean = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    return {
        "object": "block",
        "type": "heading_2",
        "heading_2": {"rich_text": [{"type": "text", "text": {"content": clean}}]}
    }


def make_heading3(text):
    """Create a heading_3 block."""
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
    """Create a table block from list of rows (each row is a list of cell strings)."""
    width = max(len(r) for r in rows)
    table_rows = []
    for row in rows:
        cells = []
        for cell in row:
            cells.append(parse_rich_text(cell.strip()))
        # Pad if needed
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


def parse_markdown(md_text):
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
            # Could be single-line or multi-line
            if line.strip().endswith('$$') and len(line.strip()) > 4:
                # Single line $$...$$
                latex = line.strip()[2:-2].strip()
                blocks.append(make_equation(latex))
                i += 1
                continue
            else:
                # Multi-line
                eq_lines = []
                i += 1
                while i < len(lines) and not lines[i].strip().startswith('$$'):
                    eq_lines.append(lines[i])
                    i += 1
                i += 1  # skip closing $$
                latex = '\n'.join(eq_lines).strip()
                blocks.append(make_equation(latex))
                continue

        # Image: ![alt](path)
        img_match = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', line.strip())
        if img_match:
            alt = img_match.group(1)
            path = img_match.group(2)
            url = FIGURE_URLS.get(path, path)
            # Check if next non-empty line is an italic caption
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

        # Table: detect | ... | ... |
        if '|' in line and i + 1 < len(lines) and re.match(r'^\|[-|: ]+\|$', lines[i + 1].strip()):
            # Parse table header
            rows = []
            header_cells = [c.strip() for c in line.strip().strip('|').split('|')]
            rows.append(header_cells)
            i += 2  # skip header + separator
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

        # Regular paragraph - may span multiple lines
        para_lines = [line]
        i += 1
        while i < len(lines):
            next_line = lines[i]
            # Stop at empty lines, headings, dividers, code, images, lists, tables, equations
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
        # Notion has a 2000-char limit per rich text block
        if len(full_para) > 1900:
            # Split into sentences
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
    """Create a Notion page and add blocks in batches of 100."""
    first_batch = blocks[:100]
    remaining = blocks[100:]

    data = {
        "parent": {"page_id": PARENT_PAGE_ID},
        "icon": {"type": "emoji", "emoji": "🖥️"},
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
    with open("/Users/raj/Desktop/Course_Creator/output/articles/5d-parallelism-gpu-programming/final.md") as f:
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
    blocks = parse_markdown(md_body)
    print(f"Generated {len(blocks)} Notion blocks")

    url = create_page("5D Parallelism: How We Train Models Too Large for Any Single GPU", blocks)
    if url:
        print(f"\nPublished to: {url}")


if __name__ == "__main__":
    main()
