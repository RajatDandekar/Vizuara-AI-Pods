import { NextRequest, NextResponse } from 'next/server';
import { getAdminUser } from '@/lib/admin';
import { getProject, saveArticleDraft, updateProject, parseFigurePlaceholders } from '@/lib/creator-projects';
import { createClaudeSSEStreamWithAccumulation } from '@/lib/claude-stream';
import { getArticleSystemPrompt, getArticlePrompt } from '@/lib/creator-prompts';
import type { FigureState } from '@/types/creator';

export const dynamic = 'force-dynamic';
export const maxDuration = 300;

export async function POST(request: NextRequest) {
  const admin = await getAdminUser();
  if (!admin) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const { projectId, outline } = await request.json();
  if (!projectId || !outline) {
    return NextResponse.json(
      { error: 'Missing projectId or outline' },
      { status: 400 }
    );
  }

  const project = await getProject(projectId);
  if (!project) {
    return NextResponse.json({ error: 'Project not found' }, { status: 404 });
  }

  const systemPrompt = await getArticleSystemPrompt();
  const userPrompt = getArticlePrompt(project.concept, outline);

  return createClaudeSSEStreamWithAccumulation(
    systemPrompt,
    userPrompt,
    async (fullText) => {
      await saveArticleDraft(projectId, fullText);

      // Auto-extract figure placeholders
      const placeholders = parseFigurePlaceholders(fullText);
      const figures: FigureState[] = placeholders.map((p) => ({
        id: p.id,
        description: p.description,
        caption: p.caption,
        method: 'paperbanana' as const,
        status: 'pending' as const,
      }));

      await updateProject(projectId, { figures });
    },
    { maxTokens: 16000 }
  );
}
