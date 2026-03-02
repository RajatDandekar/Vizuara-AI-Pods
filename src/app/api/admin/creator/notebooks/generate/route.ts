import { NextRequest, NextResponse } from 'next/server';
import { getAdminUser } from '@/lib/admin';
import { getProject, updateProject } from '@/lib/creator-projects';
import { createClaudeSSEStreamWithAccumulation } from '@/lib/claude-stream';
import { getNotebookPrompt } from '@/lib/creator-prompts';
import type { NotebookState } from '@/types/creator';

export const dynamic = 'force-dynamic';
export const maxDuration = 300;

export async function POST(request: NextRequest) {
  const admin = await getAdminUser();
  if (!admin) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  const { projectId, conceptId, title, objective, topics } = await request.json();
  const project = await getProject(projectId);
  if (!project) return NextResponse.json({ error: 'Project not found' }, { status: 404 });

  const systemPrompt =
    'You are an expert AI/ML educator creating Google Colab teaching notebooks for the Vizuara publication. ' +
    'You write in a warm, conversational professor tone using "we" and "Let us...". ' +
    'You ground every concept in first principles and build incrementally. ' +
    'Output the notebook as a markdown file where ```python blocks become code cells and everything else becomes markdown cells. ' +
    'Use #%% on its own line to create explicit cell breaks in markdown sections.';
  const userPrompt = getNotebookPrompt(
    title,
    objective,
    topics,
    project.articleDraft || ''
  );

  return createClaudeSSEStreamWithAccumulation(
    systemPrompt,
    userPrompt,
    async (fullText) => {
      // Update notebook state — markdown stored in JSONB, no filesystem
      const existing = project.notebooks.find((n: NotebookState) => n.id === conceptId);
      const notebook: NotebookState = {
        id: conceptId,
        title,
        objective,
        status: 'ready',
        markdownContent: fullText,
      };

      const notebooks = existing
        ? project.notebooks.map((n: NotebookState) =>
            n.id === conceptId ? notebook : n
          )
        : [...project.notebooks, notebook];

      await updateProject(projectId, { notebooks });
    },
    { maxTokens: 32000 }
  );
}
