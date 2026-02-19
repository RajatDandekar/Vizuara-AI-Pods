'use client';

import { useMemo } from 'react';
import MarkdownRenderer from '@/components/ui/MarkdownRenderer';

interface ArticleReaderProps {
  content: string;
  figureUrls: Record<string, string>;
  courseSlug: string;
}

export default function ArticleReader({ content, figureUrls, courseSlug }: ArticleReaderProps) {
  const processedContent = useMemo(() => {
    let processed = content;

    // Replace local figure paths with Drive URLs if available
    for (const [key, driveUrl] of Object.entries(figureUrls)) {
      // Keys may be short ("figure_1") or full paths ("figures/figure_1.png")
      const fullPath = key.includes('/') ? key : `figures/${key}.png`;
      processed = processed.replaceAll(`(${fullPath})`, `(${driveUrl})`);
    }

    // Replace remaining relative figure paths with public directory paths
    processed = processed.replace(
      /\(figures\/(figure_\d+\.png)\)/g,
      `(/courses/${courseSlug}/figures/$1)`
    );

    // Replace equation image paths with public directory paths
    processed = processed.replace(
      /\(equations\/(eq_\d+\.png)\)/g,
      `(/courses/${courseSlug}/equations/$1)`
    );

    // Remove {{FIGURE: ...}} placeholders if any remain (from draft.md format)
    processed = processed.replace(/\{\{FIGURE:[^}]*\}\}/g, '');

    // Remove {{EQUATION: ...}} placeholders — the actual $$ blocks are separate
    processed = processed.replace(/\{\{EQUATION:[^}]*\}\}/g, '');

    return processed;
  }, [content, figureUrls, courseSlug]);

  return (
    <article className="max-w-none">
      <MarkdownRenderer content={processedContent} />
    </article>
  );
}
