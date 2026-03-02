import { NextRequest, NextResponse } from 'next/server';
import { getAdminUser } from '@/lib/admin';
import { getProject, updateProject, downloadArtifact, uploadArtifact } from '@/lib/creator-projects';
import { injectNarration } from '@/lib/narration-injector';
import { createZipFromBuffers } from '@/lib/zip-utils';
import { uploadToDrive } from '@/lib/google-drive';
import type { NarrationSegment } from '@/types/creator';

export const dynamic = 'force-dynamic';
export const maxDuration = 120;

export async function POST(request: NextRequest) {
  const admin = await getAdminUser();
  if (!admin) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  const { projectId, notebookId } = await request.json();
  const project = await getProject(projectId);
  if (!project) return NextResponse.json({ error: 'Project not found' }, { status: 404 });

  // Get segments with audio for this notebook
  const segments = (project.narration?.segments || []).filter(
    (s: NarrationSegment) => s.notebookId === notebookId && s.audioStoragePath
  );

  if (segments.length === 0) {
    return NextResponse.json({ error: 'No audio segments found for this notebook' }, { status: 400 });
  }

  try {
    // Step 1: Download all audio segments from Supabase Storage
    const audioFiles: Array<{ name: string; buffer: Buffer }> = [];
    for (const seg of segments) {
      const buffer = await downloadArtifact(projectId, `narration/${seg.id}.mp3`);
      audioFiles.push({ name: `${seg.id}.mp3`, buffer });
    }

    // Step 2: Create zip from buffers (no filesystem)
    const zipBuffer = await createZipFromBuffers(audioFiles);

    // Step 3: Upload zip to Google Drive
    const zipName = `${notebookId}_narration.zip`;
    const { fileId: driveFileId } = await uploadToDrive(
      zipName,
      zipBuffer,
      'application/zip'
    );

    // Step 4: Download the notebook .ipynb from Supabase Storage
    const notebook = project.notebooks.find((n) => n.id === notebookId);
    if (!notebook?.ipynbStoragePath) {
      return NextResponse.json({ error: 'Notebook not converted yet' }, { status: 400 });
    }

    const ipynbBuffer = await downloadArtifact(projectId, `notebooks/${notebookId}.ipynb`);
    const notebookJson = JSON.parse(ipynbBuffer.toString('utf-8'));

    // Step 5: Inject narration cells using TypeScript injector
    const injectedNotebook = injectNarration(notebookJson, segments, driveFileId);

    // Step 6: Upload modified notebook back to Supabase Storage
    await uploadArtifact(
      projectId,
      `notebooks/${notebookId}.ipynb`,
      Buffer.from(JSON.stringify(injectedNotebook, null, 1), 'utf-8'),
      'application/json'
    );

    // Update narration state
    await updateProject(projectId, {
      narration: {
        ...project.narration!,
        zipDriveFileId: driveFileId,
      },
    });

    return NextResponse.json({ success: true, driveFileId });
  } catch (err) {
    console.error('[inject_narration error]', err);
    return NextResponse.json(
      { error: err instanceof Error ? err.message : 'Injection failed' },
      { status: 500 }
    );
  }
}
