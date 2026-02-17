#!/usr/bin/env python3
"""
Generate Excalidraw-style figures and export to PNG.

Pipeline:
1. Clear the Excalidraw canvas
2. Push elements via REST API
3. Screenshot the canvas via Puppeteer
4. Crop/save as PNG

Requires:
- Excalidraw server running (node /tmp/mcp_excalidraw/dist/server.js)
- Puppeteer installed (npm install -g puppeteer)

Usage:
    python generate_excalidraw.py \
      --elements elements.json \
      --output output/figure.png \
      [--server http://localhost:3100]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import requests


DEFAULT_SERVER = "http://localhost:3100"

# Virgil font character width estimates (px per char)
# Headless Puppeteer underestimates Virgil by ~20%, so these are inflated
# to prevent text clipping in rendered PNG output.
CHAR_WIDTH = {14: 12, 16: 13, 20: 17, 22: 18, 24: 20}
DEFAULT_CHAR_WIDTH = 17  # for unlisted font sizes
H_PAD = 80   # horizontal padding inside boxes (was 60, increased for safety)
V_PAD = 30   # vertical padding
LINE_HEIGHT = 28


def validate_elements(elements: list) -> list:
    """Validate and auto-fix element definitions.

    - Widens boxes that are too narrow for their label text
    - Forces roughness=0 on all arrows
    - Prints warnings for any fixes applied
    """
    fixes = 0
    for el in elements:
        el_type = el.get("type", "")

        # Force straight arrows
        if el_type == "arrow":
            if el.get("roughness", 1) != 0:
                el["roughness"] = 0
                fixes += 1
            continue

        # Check labels fit in shapes
        label = el.get("label")
        if not label or "text" not in label:
            continue

        text = label["text"]
        font_size = label.get("fontSize", el.get("fontSize", 20))
        char_w = CHAR_WIDTH.get(font_size, DEFAULT_CHAR_WIDTH)

        lines = text.split("\n")
        max_line_len = max(len(line) for line in lines)

        min_width = max_line_len * char_w + H_PAD
        min_height = len(lines) * LINE_HEIGHT + V_PAD

        # Diamonds and ellipses auto-shrink text — only warn, don't auto-fix
        if el_type in ("diamond", "ellipse"):
            cur_w = el.get("width", 0)
            cur_h = el.get("height", 0)
            if cur_w < min_width or cur_h < min_height:
                print(f"  WARN: {el_type} \"{text[:30]}\" may be tight"
                      f" ({cur_w}x{cur_h} vs {min_width}x{min_height})"
                      f" — text will auto-shrink")
            continue

        cur_w = el.get("width", 0)
        cur_h = el.get("height", 0)

        if cur_w < min_width:
            print(f"  FIX: {el_type} \"{text[:35]}\" width {cur_w} -> {min_width}")
            el["width"] = min_width
            fixes += 1

        if cur_h < min_height:
            print(f"  FIX: {el_type} \"{text[:35]}\" height {cur_h} -> {min_height}")
            el["height"] = min_height
            fixes += 1

    if fixes:
        print(f"  Applied {fixes} auto-fixes")
    else:
        print("  All elements pass validation")

    return elements


def ensure_server(server_url: str) -> bool:
    """Check if Excalidraw server is running, start it if not."""
    try:
        r = requests.get(f"{server_url}/api/elements", timeout=2)
        return r.status_code == 200
    except requests.ConnectionError:
        print("Excalidraw server not running. Starting...")
        port = server_url.split(":")[-1].rstrip("/")
        subprocess.Popen(
            ["node", "/tmp/mcp_excalidraw/dist/server.js"],
            env={**os.environ, "PORT": port},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(3)
        try:
            r = requests.get(f"{server_url}/api/elements", timeout=2)
            return r.status_code == 200
        except Exception:
            print("ERROR: Could not start Excalidraw server", file=sys.stderr)
            return False


def clear_canvas(server_url: str):
    """Delete all elements from the canvas."""
    r = requests.get(f"{server_url}/api/elements")
    data = r.json()
    for elem in data.get("elements", []):
        requests.delete(f"{server_url}/api/elements/{elem['id']}")


def push_elements(server_url: str, elements: list) -> int:
    """Push elements to the canvas. Returns count of elements created."""
    # Push in small batches to avoid server errors
    total = 0
    batch_size = 8
    for i in range(0, len(elements), batch_size):
        batch = elements[i:i + batch_size]
        r = requests.post(
            f"{server_url}/api/elements/batch",
            json={"elements": batch},
            headers={"Content-Type": "application/json"},
        )
        if r.status_code == 200:
            result = r.json()
            total += result.get("count", len(batch))
        else:
            # Fall back to individual creation
            for elem in batch:
                r2 = requests.post(
                    f"{server_url}/api/elements",
                    json=elem,
                    headers={"Content-Type": "application/json"},
                )
                if r2.status_code == 200:
                    total += 1
    return total


def screenshot_canvas(server_url: str, output_path: str, width: int = 1400, height: int = 900):
    """Use Puppeteer to screenshot Excalidraw with proper zoom-to-fit via React fiber."""
    puppeteer_script = f"""
const puppeteer = require('/tmp/node_modules/puppeteer');

(async () => {{
    const browser = await puppeteer.launch({{
        headless: 'new',
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    }});
    const page = await browser.newPage();
    await page.setViewport({{ width: {width}, height: {height} }});

    await page.goto('{server_url}', {{ waitUntil: 'networkidle2', timeout: 15000 }});

    // Wait for page to load
    await new Promise(r => setTimeout(r, 2000));

    // Ensure all fonts are loaded before Excalidraw measures text
    await page.evaluate(async () => {{
        if (document.fonts && document.fonts.ready) {{
            await document.fonts.ready;
            console.log('Fonts ready:', document.fonts.size, 'fonts loaded');
        }}
    }});

    // Wait a bit more for Excalidraw to receive elements via WebSocket
    await new Promise(r => setTimeout(r, 3000));

    // Find Excalidraw API, refresh to recalculate text with loaded fonts, then zoom-to-fit
    const zoomOk = await page.evaluate(async () => {{
        const container = document.getElementById('root');
        if (!container) return false;
        const fiberKey = Object.keys(container).find(k => k.startsWith('__reactContainer'));
        if (!fiberKey) return false;

        let fiber = container[fiberKey];
        let api = null;
        const visited = new Set();

        function walk(node, depth) {{
            if (!node || depth > 30 || visited.has(node) || api) return;
            visited.add(node);
            let state = node.memoizedState;
            while (state) {{
                const val = state.memoizedState;
                if (val && typeof val === 'object' && !Array.isArray(val) &&
                    typeof val.getSceneElements === 'function') {{
                    api = val;
                    return;
                }}
                state = state.next;
            }}
            walk(node.child, depth + 1);
            walk(node.sibling, depth + 1);
        }}

        walk(fiber, 0);
        if (!api) return false;

        // Fix text clipping: Excalidraw underestimates text width in headless mode.
        // Widen standalone text by 20% AND containers with bound labels by 15%.
        const elements = api.getSceneElements();
        const containerIds = new Set(
            elements.filter(el => el.type === 'text' && el.containerId)
                    .map(el => el.containerId)
        );
        const fixedElements = elements.map(el => {{
            if (el.type === 'text' && !el.containerId) {{
                return {{ ...el, width: Math.ceil(el.width * 1.2) }};
            }}
            if (containerIds.has(el.id)) {{
                return {{ ...el, width: Math.ceil(el.width * 1.15) }};
            }}
            return el;
        }});
        api.updateScene({{ elements: fixedElements }});

        // Wait for scene update
        await new Promise(r => setTimeout(r, 500));

        // Zoom to fit all elements with generous padding
        const updatedElements = api.getSceneElements();
        if (updatedElements && updatedElements.length > 0) {{
            api.scrollToContent(updatedElements, {{
                fitToViewport: true,
                viewportZoomFactor: 0.85
            }});
        }}
        return true;
    }});

    console.log('Zoom-to-fit via API:', zoomOk ? 'SUCCESS' : 'FAILED (using keyboard fallback)');

    if (!zoomOk) {{
        // Keyboard shortcut fallback: Ctrl+Shift+1 (Zoom to Fit)
        await page.keyboard.down('Control');
        await page.keyboard.down('Shift');
        await page.keyboard.press('Digit1');
        await page.keyboard.up('Shift');
        await page.keyboard.up('Control');
    }}

    // Wait for zoom animation
    await new Promise(r => setTimeout(r, 1500));

    // Hide all UI chrome
    await page.evaluate(() => {{
        document.querySelectorAll(
            '.header, .controls, .sync-controls, .status, ' +
            '[class*="toolbar"], [class*="Toolbar"], [class*="zoom"], [class*="Zoom"], ' +
            '[class*="footer"], [class*="Footer"], [class*="header"], [class*="Header"], ' +
            '[class*="menu"], [class*="Menu"], [class*="controls"], [class*="Controls"], ' +
            'button, .btn, [role="toolbar"], header, nav, footer'
        ).forEach(el => {{ el.style.display = 'none'; }});
        document.body.style.background = 'white';
    }});

    await new Promise(r => setTimeout(r, 500));

    await page.screenshot({{
        path: '{output_path}',
        type: 'png',
        fullPage: false,
        omitBackground: false
    }});

    await browser.close();
    console.log('Screenshot saved: {output_path}');
}})();
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
        f.write(puppeteer_script)
        script_path = f.name

    try:
        result = subprocess.run(
            ["node", script_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            print(f"Puppeteer error: {result.stderr}", file=sys.stderr)
            return False
        print(result.stdout.strip())
        return True
    except subprocess.TimeoutExpired:
        print("Puppeteer timed out", file=sys.stderr)
        return False
    finally:
        os.unlink(script_path)


def generate_excalidraw_figure(elements: list, output_path: str,
                                server_url: str = DEFAULT_SERVER) -> str | None:
    """Full pipeline: clear → push → screenshot → save."""
    if not ensure_server(server_url):
        return None

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print("Validating elements...")
    elements = validate_elements(elements)

    print("Clearing canvas...")
    clear_canvas(server_url)

    print(f"Pushing {len(elements)} elements...")
    count = push_elements(server_url, elements)
    print(f"  {count} elements on canvas")

    # Compute bounding box for viewport sizing — very generous for zoom-to-fit
    max_x = max((e.get("x", 0) + e.get("width", 200) for e in elements), default=1200)
    max_y = max((e.get("y", 0) + e.get("height", 50) for e in elements), default=800)
    # Add very generous padding (zoom-to-fit needs room)
    width = int(max_x + 500)
    height = int(max_y + 300)
    # Clamp to reasonable sizes
    width = max(1000, min(3000, width))
    height = max(600, min(2000, height))

    print(f"Taking screenshot ({width}x{height})...")
    if screenshot_canvas(server_url, output_path, width, height):
        # Auto-crop whitespace
        try:
            from PIL import Image, ImageChops
            img = Image.open(output_path)
            bg = Image.new(img.mode, img.size, (255, 255, 255))
            diff = ImageChops.difference(img, bg)
            bbox = diff.getbbox()
            if bbox:
                # Add 40px padding around content
                pad = 40
                crop_box = (
                    max(0, bbox[0] - pad),
                    max(0, bbox[1] - pad),
                    min(img.width, bbox[2] + pad),
                    min(img.height, bbox[3] + pad),
                )
                img = img.crop(crop_box)
                img.save(output_path)
                print(f"Cropped to {img.width}x{img.height}")
        except ImportError:
            pass  # Pillow not installed, skip cropping
        print(f"Figure saved: {output_path}")
        return output_path

    return None


def main():
    parser = argparse.ArgumentParser(description="Generate Excalidraw figure as PNG")
    parser.add_argument("--elements", required=True, help="JSON file with Excalidraw elements array")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--server", default=DEFAULT_SERVER, help="Excalidraw server URL")
    args = parser.parse_args()

    with open(args.elements) as f:
        elements = json.load(f)

    result = generate_excalidraw_figure(elements, args.output, args.server)
    if not result:
        print("ERROR: Figure generation failed", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()