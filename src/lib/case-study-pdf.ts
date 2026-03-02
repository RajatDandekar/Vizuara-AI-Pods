/**
 * Case study markdown → PDF conversion using Puppeteer + KaTeX.
 * TypeScript port of case_study_to_pdf.py
 *
 * Uses @sparticuz/chromium + puppeteer-core for Vercel compatibility.
 * LaTeX equations rendered via KaTeX instead of matplotlib.
 */

const CSS = `
@page {
  size: A4;
  margin: 1.8cm 2.0cm 1.8cm 2.0cm;
}

body {
  font-family: 'Georgia', 'Times New Roman', serif;
  font-size: 11pt;
  line-height: 1.5;
  color: #1a1a1a;
  max-width: 100%;
}

h1 {
  font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
  font-size: 24pt;
  font-weight: 700;
  color: #111;
  margin-top: 0;
  margin-bottom: 0.2em;
  line-height: 1.2;
}

h2 {
  font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
  font-size: 17pt;
  font-weight: 700;
  color: #222;
  margin-top: 1.2em;
  margin-bottom: 0.4em;
  border-bottom: 1px solid #e8e8e8;
  padding-bottom: 0.2em;
}

h3 {
  font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
  font-size: 13pt;
  font-weight: 600;
  color: #333;
  margin-top: 1.0em;
  margin-bottom: 0.3em;
}

h4 {
  font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
  font-size: 11.5pt;
  font-weight: 600;
  color: #444;
  margin-top: 0.8em;
  margin-bottom: 0.2em;
}

p { margin-bottom: 0.5em; text-align: left; }
strong { font-weight: 700; }
em { font-style: italic; }

table {
  border-collapse: collapse;
  width: 100%;
  margin: 0.6em 0;
  font-size: 9.5pt;
}

th {
  background: #f0f2f5;
  font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
  font-weight: 600;
  text-align: left;
  padding: 0.5em 0.7em;
  border: 1px solid #d0d0d0;
}

td {
  padding: 0.4em 0.7em;
  border: 1px solid #d0d0d0;
  vertical-align: top;
}

tr:nth-child(even) td { background: #fafafa; }

pre {
  background: #f6f8fa;
  border: 1px solid #e1e4e8;
  border-radius: 6px;
  padding: 0.8em;
  font-size: 8.5pt;
  line-height: 1.45;
  overflow-x: auto;
  margin: 0.5em 0;
}

code {
  font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
  font-size: 8.5pt;
  background: #f0f2f4;
  padding: 0.1em 0.3em;
  border-radius: 3px;
}

pre code {
  background: none;
  padding: 0;
  border-radius: 0;
  font-size: inherit;
}

blockquote {
  border-left: 4px solid #4a90d9;
  margin: 0.6em 0;
  padding: 0.4em 1em;
  background: #f8f9fb;
  color: #444;
}

hr { border: none; border-top: 1px solid #e0e0e0; margin: 1.5em 0; }
ol, ul { margin-bottom: 0.5em; padding-left: 1.5em; }
li { margin-bottom: 0.25em; }

.math-block {
  text-align: center;
  margin: 0.6em 0;
  overflow-x: auto;
}

.footer {
  position: fixed;
  bottom: 0;
  width: 100%;
  text-align: center;
  font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
  font-size: 9pt;
  color: #999;
}
`;

/**
 * Simple markdown → HTML conversion for case study content.
 * Handles headings, bold, italic, code blocks, tables, links, blockquotes,
 * and LaTeX math (via KaTeX-style spans).
 */
function markdownToHtml(md: string): string {
  let html = md;

  // Protect code blocks from math processing
  const codeBlocks: string[] = [];
  html = html.replace(/```[\w]*\n[\s\S]*?```/g, (match) => {
    codeBlocks.push(match);
    return `__CODE_BLOCK_${codeBlocks.length - 1}__`;
  });
  html = html.replace(/`[^`]+`/g, (match) => {
    codeBlocks.push(match);
    return `__CODE_BLOCK_${codeBlocks.length - 1}__`;
  });

  // Block math: $$...$$
  html = html.replace(/\$\$([\s\S]*?)\$\$/g, (_match, latex: string) => {
    const escaped = latex.trim().replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    return `<div class="math-block" data-katex="${escaped}"></div>`;
  });

  // Inline math: $...$
  // Skip currency patterns
  html = html.replace(/(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)/g, (_match, latex: string) => {
    const trimmed = latex.trim();
    if (!trimmed) return _match;
    if (/^\d[\d,]*\.?\d*\s*[KkMmBbTt%]/i.test(trimmed)) return _match;
    const escaped = trimmed.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    return `<span class="math-inline" data-katex="${escaped}"></span>`;
  });

  // Restore code blocks
  for (let i = 0; i < codeBlocks.length; i++) {
    html = html.replace(`__CODE_BLOCK_${i}__`, codeBlocks[i]);
  }

  // Code blocks
  html = html.replace(/```[\w]*\n([\s\S]*?)```/g, '<pre><code>$1</code></pre>');

  // Tables
  html = html.replace(
    /^(\|.+\|)\n(\|[\s:-]+\|)\n((?:\|.+\|\n?)+)/gm,
    (_match, header: string, _separator: string, body: string) => {
      const headerCells = header.split('|').filter((c: string) => c.trim()).map((c: string) => `<th>${c.trim()}</th>`).join('');
      const bodyRows = body.trim().split('\n').map((row: string) => {
        const cells = row.split('|').filter((c: string) => c.trim()).map((c: string) => `<td>${c.trim()}</td>`).join('');
        return `<tr>${cells}</tr>`;
      }).join('\n');
      return `<table><thead><tr>${headerCells}</tr></thead><tbody>${bodyRows}</tbody></table>`;
    }
  );

  // Headings
  html = html.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

  // Bold and italic
  html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

  // Links
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');

  // Blockquotes
  html = html.replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>');

  // Horizontal rules
  html = html.replace(/^---$/gm, '<hr>');

  // Unordered lists
  html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
  html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');

  // Ordered lists
  html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

  // Paragraphs
  html = html
    .split('\n\n')
    .map((block) => {
      const trimmed = block.trim();
      if (!trimmed) return '';
      if (trimmed.startsWith('<')) return trimmed;
      return `<p>${trimmed.replace(/\n/g, '<br>')}</p>`;
    })
    .join('\n');

  return html;
}

/**
 * Generate a PDF buffer from case study markdown content.
 *
 * Uses Puppeteer with @sparticuz/chromium for Vercel serverless compatibility.
 * KaTeX renders math equations in the browser before PDF generation.
 */
export async function generateCaseStudyPdf(
  markdownContent: string
): Promise<Buffer> {
  // Dynamic imports for edge compatibility
  const chromium = await import('@sparticuz/chromium');
  const puppeteer = await import('puppeteer-core');

  const htmlBody = markdownToHtml(markdownContent);

  const fullHtml = `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
  <style>${CSS}</style>
</head>
<body>
  ${htmlBody}
  <script>
    document.querySelectorAll('[data-katex]').forEach(el => {
      const latex = el.getAttribute('data-katex');
      const displayMode = el.classList.contains('math-block');
      katex.render(latex, el, { displayMode, throwOnError: false });
    });
  </script>
</body>
</html>`;

  const browser = await puppeteer.default.launch({
    args: chromium.default.args,
    defaultViewport: { width: 1920, height: 1080 },
    executablePath: await chromium.default.executablePath(),
    headless: true,
  });

  try {
    const page = await browser.newPage();
    await page.setContent(fullHtml, { waitUntil: 'networkidle0' });

    // Wait for KaTeX to render
    await page.waitForFunction(
      () => document.querySelectorAll('[data-katex]').length === document.querySelectorAll('.katex').length,
      { timeout: 10000 }
    ).catch(() => {
      // If KaTeX doesn't render in time, proceed anyway
    });

    const pdfBuffer = await page.pdf({
      format: 'A4',
      margin: { top: '1.8cm', right: '2cm', bottom: '1.8cm', left: '2cm' },
      printBackground: true,
    });

    return Buffer.from(pdfBuffer);
  } finally {
    await browser.close();
  }
}
