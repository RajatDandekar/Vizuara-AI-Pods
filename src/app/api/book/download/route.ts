import { NextResponse } from 'next/server';
import { getAuthUser, checkEnrollment } from '@/lib/auth';
import { readFile } from 'fs/promises';
import { join } from 'path';

export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    const user = await getAuthUser();
    if (!user) {
      return NextResponse.json({ error: 'Authentication required' }, { status: 401 });
    }

    const enrolled = await checkEnrollment(user.id);
    if (!enrolled) {
      return NextResponse.json({ error: 'Subscription required' }, { status: 403 });
    }

    const pdfPath = join(process.cwd(), 'content', 'book', 'Vizuara-Visual-AI-Book.pdf');
    const pdfBuffer = await readFile(pdfPath);

    return new NextResponse(pdfBuffer, {
      headers: {
        'Content-Type': 'application/pdf',
        'Content-Disposition': 'attachment; filename="Vizuara-Visual-AI-Book.pdf"',
        'Content-Length': pdfBuffer.byteLength.toString(),
        'Cache-Control': 'private, no-store',
      },
    });
  } catch {
    return NextResponse.json({ error: 'Download failed' }, { status: 500 });
  }
}
