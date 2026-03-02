import { NextRequest, NextResponse } from 'next/server';
import { getAdminUser } from '@/lib/admin';
import { getProject, updateProject } from '@/lib/creator-projects';
import { createClaudeSSEStreamWithAccumulation } from '@/lib/claude-stream';
import { getCaseStudyPrompt } from '@/lib/creator-prompts';

export const dynamic = 'force-dynamic';
export const maxDuration = 300;

export async function POST(request: NextRequest) {
  const admin = await getAdminUser();
  if (!admin) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  const { projectId } = await request.json();
  const project = await getProject(projectId);
  if (!project) return NextResponse.json({ error: 'Project not found' }, { status: 404 });

  const systemPrompt =
    'You are an expert AI/ML educator and industry ML consultant creating real-world case studies for the Vizuara publication. ' +
    'You create case studies that feel like they came from a real company\'s ML team, not a classroom assignment. ' +
    'You root every technical choice in first principles and explain WHY, not just WHAT. ' +
    'You use real, relevant datasets — never toy data like MNIST or CIFAR-10. ' +
    'You write in a professional but accessible tone.';

  const userPrompt = getCaseStudyPrompt(
    project.concept,
    project.articleDraft || ''
  );

  return createClaudeSSEStreamWithAccumulation(
    systemPrompt,
    userPrompt,
    async (fullText) => {
      // Store content in JSONB, no filesystem
      await updateProject(projectId, {
        caseStudy: {
          status: 'ready',
          content: fullText,
        },
      });
    },
    { maxTokens: 16000 }
  );
}
