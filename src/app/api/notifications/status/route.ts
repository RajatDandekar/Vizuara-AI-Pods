import { NextRequest, NextResponse } from 'next/server';
import { getDb } from '@/lib/db';
import { getAuthUser } from '@/lib/auth';

export const dynamic = 'force-dynamic';

export async function GET(request: NextRequest) {
  try {
    const user = await getAuthUser();
    if (!user) {
      return NextResponse.json({ subscribed: false });
    }

    const courseSlug = request.nextUrl.searchParams.get('courseSlug');
    if (!courseSlug) {
      return NextResponse.json({ error: 'courseSlug is required' }, { status: 400 });
    }

    const supabase = getDb();

    const { data: existing } = await supabase
      .from('notifications')
      .select('id')
      .eq('user_id', user.id)
      .eq('course_slug', courseSlug)
      .maybeSingle();

    return NextResponse.json({ subscribed: !!existing });
  } catch {
    return NextResponse.json({ subscribed: false });
  }
}
