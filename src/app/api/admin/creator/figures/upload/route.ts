import { NextRequest, NextResponse } from 'next/server';
import { getAdminUser } from '@/lib/admin';
import { getProject, updateProject, uploadArtifact } from '@/lib/creator-projects';
import type { FigureState } from '@/types/creator';

export const dynamic = 'force-dynamic';

export async function POST(request: NextRequest) {
  const admin = await getAdminUser();
  if (!admin) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  const formData = await request.formData();
  const projectId = formData.get('projectId') as string;
  const figureId = formData.get('figureId') as string;
  const file = formData.get('file') as File;

  if (!projectId || !figureId || !file) {
    return NextResponse.json({ error: 'Missing required fields' }, { status: 400 });
  }

  const project = await getProject(projectId);
  if (!project) return NextResponse.json({ error: 'Project not found' }, { status: 404 });

  // Save file to Supabase Storage
  const buffer = Buffer.from(await file.arrayBuffer());
  const storagePath = await uploadArtifact(
    projectId,
    `figures/${figureId}.png`,
    buffer,
    'image/png'
  );

  // Update figure status
  const figures = project.figures.map((f: FigureState) =>
    f.id === figureId
      ? { ...f, status: 'ready' as const, storagePath }
      : f
  );
  await updateProject(projectId, { figures });

  return NextResponse.json({ success: true, storagePath });
}
