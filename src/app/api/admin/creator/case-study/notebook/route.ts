import { NextRequest, NextResponse } from 'next/server';
import { getAdminUser } from '@/lib/admin';
import { getProject, updateProject } from '@/lib/creator-projects';
import { createClaudeSSEStreamWithAccumulation } from '@/lib/claude-stream';
import { getCaseStudyNotebookPrompt } from '@/lib/creator-prompts';

export const dynamic = 'force-dynamic';
export const maxDuration = 300;

export async function POST(request: NextRequest) {
  const admin = await getAdminUser();
  if (!admin) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  const { projectId } = await request.json();
  const project = await getProject(projectId);
  if (!project) return NextResponse.json({ error: 'Project not found' }, { status: 404 });

  if (!project.caseStudy?.content) {
    return NextResponse.json({ error: 'Case study content required' }, { status: 400 });
  }

  const systemPrompt =
    'You are an expert AI/ML educator creating a Google Colab teaching notebook for the Vizuara publication. ' +
    'You write in a warm, conversational professor tone using "we" and "Let us...". ' +
    'You ground every concept in first principles and build incrementally. ' +
    'Output the notebook as a markdown file where ```python blocks become code cells and everything else becomes markdown cells. ' +
    'Use #%% on its own line to create explicit cell breaks in markdown sections.';

  const userPrompt = getCaseStudyNotebookPrompt(
    project.caseStudy.content,
    project.concept
  );

  return createClaudeSSEStreamWithAccumulation(
    systemPrompt,
    userPrompt,
    async (fullText) => {
      // Store in JSONB, no filesystem
      await updateProject(projectId, {
        caseStudy: {
          status: project.caseStudy!.status,
          content: project.caseStudy!.content,
          pdfStoragePath: project.caseStudy!.pdfStoragePath,
          notebookMarkdown: fullText,
          notebookStatus: 'ready',
        },
      });
    },
    { maxTokens: 32000 }
  );
}
