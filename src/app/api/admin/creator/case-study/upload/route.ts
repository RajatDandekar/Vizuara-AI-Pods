import { NextRequest, NextResponse } from 'next/server';
import { getAdminUser } from '@/lib/admin';
import { getProject, updateProject, downloadArtifact } from '@/lib/creator-projects';
import { uploadToDrive, getColabUrl } from '@/lib/google-drive';
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
  if (!cs?.notebookIpynbStoragePath) {
    return NextResponse.json({ error: 'Notebook not converted yet' }, { status: 400 });
  }

  try {
    // Download from Supabase Storage
    const ipynbBuffer = await downloadArtifact(
      projectId,
      'case-study/case_study_notebook.ipynb'
    );

    // Upload to Google Drive via API
    const { fileId } = await uploadToDrive(
      'case_study_notebook.ipynb',
      ipynbBuffer,
      'application/x-ipynb+json'
    );

    const colabUrl = getColabUrl(fileId);

    const latest = await getProject(projectId);
    await updateProject(projectId, {
      caseStudy: mergeCaseStudy(latest!.caseStudy!, {
        notebookColabUrl: colabUrl,
        notebookDriveFileId: fileId,
        notebookStatus: 'uploaded',
      }),
    });

    return NextResponse.json({ colabUrl, fileId });
  } catch (err) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : 'Upload failed' },
      { status: 500 }
    );
  }
}
