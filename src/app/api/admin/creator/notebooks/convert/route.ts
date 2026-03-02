import { NextRequest, NextResponse } from 'next/server';
import { getAdminUser } from '@/lib/admin';
import { getProject, updateProject, createJob, updateJob, uploadArtifact } from '@/lib/creator-projects';
import { generateNotebook } from '@/lib/notebook-converter';
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
  if (!notebook?.markdownContent) {
    return NextResponse.json({ error: 'Notebook content not found' }, { status: 404 });
  }

  const job = await createJob(projectId, 'notebook-convert');

  // Update status
  const notebooks = project.notebooks.map((n: NotebookState) =>
    n.id === conceptId ? { ...n, status: 'converting' as const } : n
  );
  await updateProject(projectId, { notebooks });

  // Run conversion in background
  (async () => {
    try {
      await updateJob(projectId, job.id, {
        status: 'running',
        progress: 50,
        message: `Converting ${conceptId} to .ipynb...`,
      });

      // Use TypeScript converter directly
      const ipynbJson = generateNotebook(
        notebook.title || 'Vizuara Notebook',
        notebook.markdownContent!
      );

      // Upload to Supabase Storage
      const storagePath = await uploadArtifact(
        projectId,
        `notebooks/${conceptId}.ipynb`,
        Buffer.from(ipynbJson, 'utf-8'),
        'application/json'
      );

      await updateJob(projectId, job.id, {
        status: 'complete',
        progress: 100,
        result: { storagePath },
      });

      const updatedProject = await getProject(projectId);
      if (updatedProject) {
        const updatedNotebooks = updatedProject.notebooks.map((n: NotebookState) =>
          n.id === conceptId
            ? { ...n, status: 'ready' as const, ipynbStoragePath: storagePath }
            : n
        );
        await updateProject(projectId, { notebooks: updatedNotebooks });
      }
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : 'Conversion failed';
      await updateJob(projectId, job.id, { status: 'error', error: errMsg });

      const updatedProject = await getProject(projectId);
      if (updatedProject) {
        const updatedNotebooks = updatedProject.notebooks.map((n: NotebookState) =>
          n.id === conceptId
            ? { ...n, status: 'error' as const, error: errMsg }
            : n
        );
        await updateProject(projectId, { notebooks: updatedNotebooks });
      }
    }
  })();

  return NextResponse.json({ jobId: job.id });
}
