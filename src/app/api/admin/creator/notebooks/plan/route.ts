import { NextRequest, NextResponse } from 'next/server';
import { getAdminUser } from '@/lib/admin';
import { getProject, updateProject } from '@/lib/creator-projects';
import { splitArticleIntoConcepts } from '@/lib/concept-splitter';
import type { ConceptPlan, ConceptPlanItem } from '@/types/creator';

export const dynamic = 'force-dynamic';
export const maxDuration = 60;

export async function POST(request: NextRequest) {
  const admin = await getAdminUser();
  if (!admin) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  const { projectId } = await request.json();
  const project = await getProject(projectId);
  if (!project) return NextResponse.json({ error: 'Project not found' }, { status: 404 });

  if (!project.articleDraft) {
    return NextResponse.json({ error: 'Article draft required' }, { status: 400 });
  }

  try {
    // Use TypeScript concept splitter directly
    const rawConcepts = splitArticleIntoConcepts(project.articleDraft);

    // Transform to ConceptPlan format
    const concepts: ConceptPlanItem[] = rawConcepts.map((c) => ({
      id: `${String(c.index).padStart(2, '0')}_${c.concept_name.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/_+$/, '').slice(0, 40)}`,
      title: c.concept_name,
      objective: c.description || c.suggested_final_output,
      topics: [...c.subsections.slice(0, 5), ...c.key_terms.slice(0, 3)],
    }));

    const conceptPlan: ConceptPlan = { concepts };
    await updateProject(projectId, { conceptPlan, conceptPlanApproved: false });

    return NextResponse.json({ conceptPlan });
  } catch (err) {
    console.error('[concept_splitter error]', err);
    return NextResponse.json(
      { error: err instanceof Error ? err.message : 'Failed to generate concept plan' },
      { status: 500 }
    );
  }
}
