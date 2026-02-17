#!/usr/bin/env python3
"""
Classify figure descriptions as 'flowchart' or 'illustration'.

Flowchart figures are rendered as Mermaid diagrams (native in Notion).
Illustration figures are rendered as PNGs via PaperBanana/Gemini.

Usage:
    python classify_figure.py --description "A pipeline showing X -> Y -> Z"
    # Output: flowchart

    python classify_figure.py --description "A bar chart comparing model performance"
    # Output: illustration
"""

import argparse
import re


# Keywords that strongly indicate a flowchart/diagram
FLOWCHART_KEYWORDS = [
    r'\bflowchart\b', r'\bflow\s*chart\b', r'\bpipeline\b',
    r'\bflow\s*diagram\b', r'\bprocess\s*flow\b',
    r'\bstages?\b.*\barrow', r'\barrow.*\bstages?\b',
    r'\bstep[s]?\b.*\bflow\b', r'\bflow\b.*\bstep[s]?\b',
    r'\b(left|right)\s*panel\b.*\b(left|right)\s*panel\b',  # Two-panel comparison
    r'\btwo-panel\b', r'\bthree-panel\b', r'\bmulti-panel\b',
    r'\bcomparison\s*diagram\b',
    r'\bblock\s*diagram\b', r'\barchitecture\s*diagram\b',
    r'\bsequence\s*diagram\b',
    r'\b-+>\b',  # Arrow notation like -->
    r'\bconnected\s*by\s*(thick\s*)?(arrows?|lines?)\b',
    r'\bflow[s]?\s*(left\s*to\s*right|top\s*to\s*bottom)\b',
    r'\bvertical\s*flow\b', r'\bhorizontal\s*flow\b',
    r'\bdata\s*flow\b',
    r'\bdecision\s*(tree|diamond|box)\b',
    r'\bif\s*yes\b.*\bif\s*no\b',
    r'\bboxes?\s*(stacked|connected|linked)\b',
]

# Keywords that strongly indicate an illustration/data visualization
ILLUSTRATION_KEYWORDS = [
    r'\bbar\s*chart\b', r'\bline\s*chart\b', r'\bline\s*graph\b',
    r'\bradar\s*chart\b', r'\bspider\s*chart\b',
    r'\bheatmap\b', r'\bscatter\s*plot\b', r'\bhistogram\b',
    r'\bpie\s*chart\b', r'\barea\s*chart\b',
    r'\btraining\s*curves?\b',
    r'\by-axis\b', r'\bx-axis\b', r'\btwo\s*y-axes\b',
    r'\bplotted\b', r'\bbar\s*heights?\b',
    r'\bapproximate\s*bar\b',
    r'\billustration\b', r'\bphotorealistic\b',
    r'\bconcept\s*art\b', r'\bscene\b',
    r'\bannotation\s*line\b.*\bdashed\b',
]

# Structural signals: descriptions that talk about boxes + arrows = flowchart
STRUCTURAL_SIGNALS = [
    (r'\bbox(es)?\b', r'\barrow[s]?\b'),  # boxes AND arrows
    (r'\brectangle\b', r'\barrow\b'),
    (r'\bstage\s*\d\b', r'\bstage\s*\d\b'),  # Multiple numbered stages
]


def classify_figure(description: str) -> str:
    """
    Classify a figure description as 'flowchart' or 'illustration'.

    Returns:
        'flowchart' — render as Mermaid in Notion
        'illustration' — render as PNG via PaperBanana
    """
    desc_lower = description.lower()

    flowchart_score = 0
    illustration_score = 0

    # Check flowchart keywords
    for pattern in FLOWCHART_KEYWORDS:
        if re.search(pattern, desc_lower):
            flowchart_score += 1

    # Check illustration keywords
    for pattern in ILLUSTRATION_KEYWORDS:
        if re.search(pattern, desc_lower):
            illustration_score += 1

    # Check structural signals (require BOTH parts to match)
    for pat_a, pat_b in STRUCTURAL_SIGNALS:
        if re.search(pat_a, desc_lower) and re.search(pat_b, desc_lower):
            flowchart_score += 2

    # Strong override: if it mentions axes or plotted data, it's illustration
    if re.search(r'\b(y-axis|x-axis|plotted|bar heights)\b', desc_lower):
        illustration_score += 3

    # Strong override: if it has --> or "flow left to right" with stages
    if re.search(r'stage\s*\d.*stage\s*\d.*stage\s*\d', desc_lower):
        flowchart_score += 3

    if flowchart_score > illustration_score:
        return 'flowchart'
    elif illustration_score > flowchart_score:
        return 'illustration'
    else:
        # Default: if it mentions panels or comparison with structural elements, flowchart
        if re.search(r'\bpanel\b', desc_lower) and re.search(r'\b(arrow|box|flow)\b', desc_lower):
            return 'flowchart'
        # Default to illustration (PaperBanana handles most things well)
        return 'illustration'


def main():
    parser = argparse.ArgumentParser(description="Classify figure as flowchart or illustration")
    parser.add_argument("--description", required=True, help="Figure description text")
    args = parser.parse_args()

    result = classify_figure(args.description)
    print(result)


if __name__ == "__main__":
    main()