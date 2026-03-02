import { NextRequest, NextResponse } from 'next/server';
import { getAdminUser } from '@/lib/admin';
import { getProject, updateProject, uploadArtifact } from '@/lib/creator-projects';
import { generateCaseStudyPdf } from '@/lib/case-study-pdf';
import type { CaseStudyState } from '@/types/creator';

export const dynamic = 'force-dynamic';
export const maxDuration = 120;

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
  if (!cs?.content) {
    return NextResponse.json({ error: 'Case study content required' }, { status: 400 });
  }

  try {
    // Generate PDF using Puppeteer + KaTeX (no Python/WeasyPrint)
    const pdfBuffer = await generateCaseStudyPdf(cs.content);

    // Upload to Supabase Storage
    const storagePath = await uploadArtifact(
      projectId,
      'case-study/case_study.pdf',
      pdfBuffer,
      'application/pdf'
    );

    const latest = await getProject(projectId);
    await updateProject(projectId, {
      caseStudy: mergeCaseStudy(latest!.caseStudy!, { pdfStoragePath: storagePath }),
    });

    return NextResponse.json({ storagePath });
  } catch (err) {
    const errMsg = err instanceof Error ? err.message : 'PDF generation failed';
    console.error('[case_study_pdf error]', err);
    return NextResponse.json({ error: errMsg }, { status: 500 });
  }
}
