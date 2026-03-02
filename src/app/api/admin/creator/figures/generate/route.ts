import { NextRequest, NextResponse } from 'next/server';
import { getAdminUser } from '@/lib/admin';
import { getProject, updateProject, createJob, updateJob, uploadArtifact } from '@/lib/creator-projects';
import { generateFigure } from '@/lib/figure-generator';
import type { FigureState } from '@/types/creator';

export const dynamic = 'force-dynamic';
export const maxDuration = 300;

export async function POST(request: NextRequest) {
  const admin = await getAdminUser();
  if (!admin) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  const { projectId, figureId, description } = await request.json();

  const project = await getProject(projectId);
  if (!project) return NextResponse.json({ error: 'Project not found' }, { status: 404 });

  // Create a job for tracking
  const job = await createJob(projectId, 'figure-generate');

  // Update figure status
  const figures = project.figures.map((f: FigureState) =>
    f.id === figureId ? { ...f, status: 'generating' as const, jobId: job.id } : f
  );
  await updateProject(projectId, { figures });

  // Run generation in background
  (async () => {
    try {
      await updateJob(projectId, job.id, {
        status: 'running',
        progress: 10,
        message: `Generating ${figureId}...`,
      });

      const imageBuffer = await generateFigure(description);

      if (!imageBuffer) {
        throw new Error('Figure generation returned no image');
      }

      // Upload to Supabase Storage
      const storagePath = await uploadArtifact(
        projectId,
        `figures/${figureId}.png`,
        imageBuffer,
        'image/png'
      );

      await updateJob(projectId, job.id, {
        status: 'complete',
        progress: 100,
        message: 'Figure generated successfully',
        result: { storagePath },
      });

      const updatedProject = await getProject(projectId);
      if (updatedProject) {
        const updatedFigures = updatedProject.figures.map((f: FigureState) =>
          f.id === figureId
            ? { ...f, status: 'ready' as const, storagePath, method: 'paperbanana' as const }
            : f
        );
        await updateProject(projectId, { figures: updatedFigures });
      }
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : 'Figure generation failed';
      console.error(`[figure-gen] ${figureId} error:`, errMsg);
      await updateJob(projectId, job.id, {
        status: 'error',
        message: errMsg,
        error: errMsg,
      });

      const updatedProject = await getProject(projectId);
      if (updatedProject) {
        const updatedFigures = updatedProject.figures.map((f: FigureState) =>
          f.id === figureId
            ? { ...f, status: 'error' as const, error: errMsg }
            : f
        );
        await updateProject(projectId, { figures: updatedFigures });
      }
    }
  })();

  return NextResponse.json({ jobId: job.id });
}
