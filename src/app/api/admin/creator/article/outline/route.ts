import { NextRequest } from 'next/server';
import { getAdminUser } from '@/lib/admin';
import { NextResponse } from 'next/server';
import { getProject, saveOutline, updateStepStatus } from '@/lib/creator-projects';
import {
  createClaudeSSEStreamWithAccumulation,
} from '@/lib/claude-stream';
import {
  getArticleSystemPrompt,
  getOutlinePrompt,
} from '@/lib/creator-prompts';

export const dynamic = 'force-dynamic';
export const maxDuration = 120;

export async function POST(request: NextRequest) {
  const admin = await getAdminUser();
  if (!admin) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const { projectId } = await request.json();
  if (!projectId) {
    return NextResponse.json({ error: 'Missing projectId' }, { status: 400 });
  }

  const project = await getProject(projectId);
  if (!project) {
    return NextResponse.json({ error: 'Project not found' }, { status: 404 });
  }

  await updateStepStatus(projectId, 'article', 'in-progress');

  const systemPrompt = await getArticleSystemPrompt();
  const userPrompt = getOutlinePrompt(project.concept);

  return createClaudeSSEStreamWithAccumulation(
    systemPrompt,
    userPrompt,
    async (fullText) => {
      await saveOutline(projectId, fullText);
    },
    { maxTokens: 8000 }
  );
}
