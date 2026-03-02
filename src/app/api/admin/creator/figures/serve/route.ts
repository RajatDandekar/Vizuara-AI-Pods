import { NextRequest, NextResponse } from 'next/server';
import { getAdminUser } from '@/lib/admin';
import { getArtifactUrl } from '@/lib/creator-projects';

export const dynamic = 'force-dynamic';

export async function GET(request: NextRequest) {
  const admin = await getAdminUser();
  if (!admin) return new NextResponse('Unauthorized', { status: 401 });

  const projectId = request.nextUrl.searchParams.get('projectId');
  const figureId = request.nextUrl.searchParams.get('figureId');

  if (!projectId || !figureId) {
    return new NextResponse('Missing params', { status: 400 });
  }

  try {
    const signedUrl = await getArtifactUrl(
      projectId,
      `figures/${figureId}.png`
    );

    // Redirect to the signed URL
    return NextResponse.redirect(signedUrl);
  } catch {
    return new NextResponse('Not found', { status: 404 });
  }
}
