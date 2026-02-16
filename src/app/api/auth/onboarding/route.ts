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

    const { interests, experienceLevel } = await request.json();

    const supabase = getDb();

    // Update experience level and onboarding status
    await supabase
      .from('users')
      .update({
        experience_level: experienceLevel || null,
        onboarding_complete: true,
        updated_at: new Date().toISOString(),
      })
      .eq('id', user.id);

    // Clear existing interests and insert new ones
    await supabase
      .from('user_interests')
      .delete()
      .eq('user_id', user.id);

    if (interests && Array.isArray(interests) && interests.length > 0) {
      const rows = interests.map((tag: string) => ({
        id: generateId(),
        user_id: user.id,
        tag,
      }));
      await supabase.from('user_interests').insert(rows);
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Onboarding error:', error);
    return NextResponse.json(
      { error: 'Something went wrong' },
      { status: 500 }
    );
  }
}
