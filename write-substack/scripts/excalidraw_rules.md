# Excalidraw Element Rules

Rules to follow when creating Excalidraw element JSON definitions. Excalidraw uses
the hand-drawn "Virgil" font which is significantly wider than standard fonts.
Headless Puppeteer underestimates Virgil text width by ~20%.

## Text Sizing (Critical — Causes Text Clipping if Wrong)

Character width estimates for Virgil font (inflated for headless rendering):
- fontSize 14: ~12px per character
- fontSize 16: ~13px per character
- fontSize 20 (default): ~17px per character
- fontSize 22: ~18px per character
- fontSize 24: ~20px per character

### Box Width Formula

For rectangles with labels:
```
min_width = longest_line_chars × 17 + 80px padding
```

For diamonds: multiply by 1.6 (text area is smaller than bounding box)
For ellipses: multiply by 1.5

### Box Height Formula

```
min_height = num_lines × 28 + 30px padding
```

For diamonds: multiply by 1.4
For ellipses: multiply by 1.3

### ALWAYS Over-Estimate Width

It is ALWAYS better for a box to be slightly too wide than to have clipped text.
If in doubt, add 20% more width. The render pipeline will also widen containers
by 15% at render time as a safety net, but do not rely on this.

## Arrow Rules

- **Always set `roughness: 0`** on arrows to prevent slanted rendering
- Vertical arrows: `width: 0` (non-zero width causes diagonal)
- Horizontal arrows: `height: 0` (non-zero height causes diagonal)
- Arrow gap: at least 15px between source box bottom and arrow start
- Arrow length: 20-40px for connectors
- **Every arrow must connect two identifiable elements** — no floating arrows
  from/to empty space

## Labels vs Standalone Text (Critical)

**Excalidraw auto-sizes label text to fill the container.** This means:
- Bigger box = bigger text = potential overflow for long text
- Labels work well for SHORT text in moderately-sized boxes (< 350px)
- For LONG text (> 20 chars) in wide boxes (> 400px), use **standalone text elements**
  instead of labels to get explicit fontSize control

### When to use labels:
- Short labels like "Reinforce", "Correct?", "Reward Model"
- Text that fits comfortably within the box at default sizing

### When to use standalone text:
- Long text like "Problem: Solve for x in 3x + 7 = 22"
- Labels > 15 chars in boxes wider than 300px (Excalidraw scales up the font,
  causing text to clip even when the box passes width validation)
- Multi-line content where you need precise positioning
- Any time you need to control the exact font size

### Standalone text properties:
- Always set explicit `width`: `chars × char_width + 20`
- Set `fontSize` explicitly (16-18 for body text, 22 for titles)
- Position at (box_x + padding, box_y + padding) to visually sit inside the box

## Layout Rules

- Align boxes on a common center-X within each panel for clean vertical arrows
- Minimum gap between stacked boxes: 10px (after arrow)
- Parent containers: 30px margin on all sides around children
- For side-by-side panels: 30-50px gap between panels

## Multi-Line Labels

- Use `\n` for line breaks in label text
- Width determined by the LONGEST line
- Each line measured independently

## Anti-Overlap Rules (Critical)

**The auto-fix validator will widen boxes** to fit their text. You MUST account for
this when designing layouts, or boxes will collide after auto-fix.

### Pre-compute auto-fixed widths BEFORE choosing positions

For every labeled shape, calculate the auto-fixed width FIRST:
```
auto_width = longest_line_chars × 17 + 80
```
Then use that width (or larger) when computing x-positions. Never assume your
original width will survive — if it's smaller than `auto_width`, it WILL be widened.

### Minimum gap between adjacent boxes: 50px

After computing auto-fixed widths, verify:
```
box_A.x + box_A.auto_width + 50 <= box_B.x
```
for every pair of horizontally adjacent boxes.

### No arrow path may cross through a box

When designing multi-point arrows (loops, L-shapes), trace the full path and
verify that every segment stays outside all box boundaries.

Fix: route loop-back arrows at least 50px to the left/right of the leftmost/
rightmost box edge.

### Side-by-side boxes in constrained panels

When placing 2+ boxes side-by-side within a panel, verify:
```
sum(auto_fixed_widths) + gaps + left_margin + right_margin <= panel_width
```
If they don't fit, either:
1. Make the panel wider
2. Use shorter label text
3. Stack the boxes vertically instead

## Validation Checklist

Before rendering, verify:
1. Every labeled shape has width >= min_width for its text
2. Every labeled shape has height >= min_height for its lines
3. All arrows have roughness=0
4. Vertical arrows have width=0, horizontal arrows have height=0
5. No box exceeds its parent container boundaries
6. Arrow start/end coordinates align with box centers
7. **No two sibling boxes overlap** after accounting for auto-fixed widths
8. **No arrow path crosses through any box** it isn't connecting to
9. **50px minimum gap** between all horizontally adjacent boxes
10. **Every arrow connects two identifiable elements** — no floating arrows