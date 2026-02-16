import { NextRequest, NextResponse } from 'next/server';
import { getDb } from '@/lib/db';
import { getAuthUser, generateId } from '@/lib/auth';

export const dynamic = 'force-dynamic';

export async function POST(request: NextRequest) {
  try {
    const user = await getAuthUser();
    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const { courseSlug } = await request.json();
    if (!courseSlug) {
      return NextResponse.json({ error: 'courseSlug is required' }, { status: 400 });
    }

    const supabase = getDb();

    // Upsert notification (ignore if already exists)
    await supabase
      .from('notifications')
      .upsert(
        {
          id: generateId(),
          user_id: user.id,
          course_slug: courseSlug,
        },
        {
          onConflict: 'user_id,course_slug',
          ignoreDuplicates: true,
        }
      );

    return NextResponse.json({ subscribed: true });
  } catch (error) {
    console.error('Subscribe error:', error);
    return NextResponse.json({ error: 'Something went wrong' }, { status: 500 });
  }
}

export async function DELETE(request: NextRequest) {
  try {
    const user = await getAuthUser();
    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const { courseSlug } = await request.json();
    if (!courseSlug) {
      return NextResponse.json({ error: 'courseSlug is required' }, { status: 400 });
    }

    const supabase = getDb();

    await supabase
      .from('notifications')
      .delete()
      .eq('user_id', user.id)
      .eq('course_slug', courseSlug);

    return NextResponse.json({ subscribed: false });
  } catch (error) {
    console.error('Unsubscribe error:', error);
    return NextResponse.json({ error: 'Something went wrong' }, { status: 500 });
  }
}
