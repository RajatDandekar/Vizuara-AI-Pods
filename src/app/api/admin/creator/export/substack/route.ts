import { NextRequest, NextResponse } from 'next/server';
import { getAdminUser } from '@/lib/admin';
import { getProject } from '@/lib/creator-projects';
import type { FigureState } from '@/types/creator';

export const dynamic = 'force-dynamic';

export async function POST(request: NextRequest) {
  const admin = await getAdminUser();
  if (!admin) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  const { projectId } = await request.json();
  const project = await getProject(projectId);
  if (!project) return NextResponse.json({ error: 'Project not found' }, { status: 404 });

  let article = project.articleDraft || '';

  // Replace figure placeholders with Drive URLs or serve URLs
  for (const figure of project.figures) {
    const placeholder = new RegExp(
      `\\{\\{FIGURE:.*?\\}\\}`,
      ''
    );

    if (figure.driveUrl) {
      article = article.replace(
        placeholder,
        `![${figure.caption}](${figure.driveUrl})`
      );
    } else if (figure.storagePath) {
      article = article.replace(
        placeholder,
        `![${figure.caption}](/api/admin/creator/figures/serve?projectId=${projectId}&figureId=${figure.id})`
      );
    }
  }

  // Extract block equations for separate handling
  const blockEquations: { index: number; latex: string }[] = [];
  let eqIndex = 0;
  article = article.replace(/\$\$([^$]+)\$\$/g, (_match, latex) => {
    eqIndex++;
    blockEquations.push({ index: eqIndex, latex: latex.trim() });
    return `\n\n[EQUATION ${eqIndex}: See equation list below]\n\n`;
  });

  // Convert markdown to simple HTML for Substack paste
  const html = markdownToSubstackHTML(article);

  return NextResponse.json({
    html,
    markdown: article,
    blockEquations,
    figures: project.figures.filter((f: FigureState) => f.status === 'ready' || f.status === 'approved'),
  });
}

function markdownToSubstackHTML(md: string): string {
  let html = md;

  // Headers
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

  // Bold and italic
  html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

  // Images
  html = html.replace(
    /!\[([^\]]*)\]\(([^)]+)\)/g,
    '<figure><img src="$2" alt="$1"><figcaption>$1</figcaption></figure>'
  );

  // Links
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');

  // Code blocks
  html = html.replace(
    /```[\w]*\n([\s\S]*?)```/g,
    '<pre><code>$1</code></pre>'
  );

  // Blockquotes
  html = html.replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>');

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
