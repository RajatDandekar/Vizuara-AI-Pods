import { NextRequest, NextResponse } from 'next/server';
import { getAdminUser } from '@/lib/admin';
import { getProject, updateProject, downloadArtifact } from '@/lib/creator-projects';
import { uploadToDrive, getColabUrl } from '@/lib/google-drive';
import type { NotebookState } from '@/types/creator';

export const dynamic = 'force-dynamic';
export const maxDuration = 60;

export async function POST(request: NextRequest) {
  const admin = await getAdminUser();
  if (!admin) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  const { projectId, conceptId } = await request.json();
  const project = await getProject(projectId);
  if (!project) return NextResponse.json({ error: 'Project not found' }, { status: 404 });

  const notebook = project.notebooks.find((n: NotebookState) => n.id === conceptId);
  if (!notebook?.ipynbStoragePath) {
    return NextResponse.json({ error: 'Notebook not converted yet' }, { status: 400 });
  }

  const filename = `${conceptId}.ipynb`;

  try {
    // Download from Supabase Storage
    const ipynbBuffer = await downloadArtifact(projectId, `notebooks/${conceptId}.ipynb`);

    // Upload to Google Drive via API
    const { fileId } = await uploadToDrive(
      filename,
      ipynbBuffer,
      'application/x-ipynb+json'
    );

    const colabUrl = getColabUrl(fileId);

    // Update notebook state
    const notebooks = project.notebooks.map((n: NotebookState) =>
      n.id === conceptId
        ? { ...n, status: 'uploaded' as const, colabUrl, driveFileId: fileId }
        : n
    );
    await updateProject(projectId, { notebooks });

    return NextResponse.json({ colabUrl, fileId });
  } catch (err) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : 'Upload failed' },
      { status: 500 }
    );
  }
}
