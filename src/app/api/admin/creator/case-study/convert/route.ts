import { NextRequest, NextResponse } from 'next/server';
import { getAdminUser } from '@/lib/admin';
import { getProject, updateProject, uploadArtifact } from '@/lib/creator-projects';
import { generateNotebook } from '@/lib/notebook-converter';
import type { CaseStudyState } from '@/types/creator';

export const dynamic = 'force-dynamic';
export const maxDuration = 60;

/** Merge updates into caseStudy, preserving required 'status' field */
function mergeCaseStudy(
  existing: CaseStudyState,
  patch: Partial<CaseStudyState>
): CaseStudyState {
  return { ...existing, ...patch };
}

export async function POST(request: NextRequest) {
  const admin = await getAdminUser();
  if (!admin) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  const { projectId } = await request.json();
  const project = await getProject(projectId);
  if (!project) return NextResponse.json({ error: 'Project not found' }, { status: 404 });

  const cs = project.caseStudy;
  if (!cs?.notebookMarkdown) {
    return NextResponse.json({ error: 'Notebook markdown not generated yet' }, { status: 400 });
  }

  try {
    await updateProject(projectId, {
      caseStudy: mergeCaseStudy(cs, { notebookStatus: 'converting' }),
    });

    // Use TypeScript converter directly
    const ipynbJson = generateNotebook(
      `${project.concept} — Case Study Notebook`,
      cs.notebookMarkdown
    );

    // Upload to Supabase Storage
    const storagePath = await uploadArtifact(
      projectId,
      'case-study/case_study_notebook.ipynb',
      Buffer.from(ipynbJson, 'utf-8'),
      'application/json'
    );

    // Re-read to get latest state
    const latest = await getProject(projectId);
    await updateProject(projectId, {
      caseStudy: mergeCaseStudy(latest!.caseStudy!, {
        notebookIpynbStoragePath: storagePath,
        notebookStatus: 'ready',
      }),
    });

    return NextResponse.json({ storagePath });
  } catch (err) {
    const errMsg = err instanceof Error ? err.message : 'Conversion failed';
    const latest = await getProject(projectId);
    if (latest?.caseStudy) {
      await updateProject(projectId, {
        caseStudy: mergeCaseStudy(latest.caseStudy, {
          notebookStatus: 'error',
          notebookError: errMsg,
        }),
      });
    }
    return NextResponse.json({ error: errMsg }, { status: 500 });
  }
}
