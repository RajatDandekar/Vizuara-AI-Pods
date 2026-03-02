import { NextRequest, NextResponse } from 'next/server';
import { getAdminUser } from '@/lib/admin';
import { listProjects, createProject } from '@/lib/creator-projects';
import type { CreateProjectRequest } from '@/types/creator';

export const dynamic = 'force-dynamic';

export async function GET() {
  const admin = await getAdminUser();
  if (!admin) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  const projects = await listProjects();
  return NextResponse.json({ projects });
}

export async function POST(request: NextRequest) {
  const admin = await getAdminUser();
  if (!admin) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  const body = (await request.json()) as CreateProjectRequest;

  if (!body.concept || !body.podTitle || !body.podSlug || !body.courseSlug) {
    return NextResponse.json(
      { error: 'Missing required fields: concept, podTitle, podSlug, courseSlug' },
      { status: 400 }
    );
  }

  const project = await createProject(body, admin.id);
  return NextResponse.json({ project }, { status: 201 });
}
