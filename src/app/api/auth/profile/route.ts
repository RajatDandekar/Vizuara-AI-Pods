import { NextRequest, NextResponse } from 'next/server';
import { getDb } from '@/lib/db';
import { getAuthUser, generateId } from '@/lib/auth';

export const dynamic = 'force-dynamic';

export async function PUT(request: NextRequest) {
  try {
    const user = await getAuthUser();
    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const { fullName, interests, experienceLevel } = await request.json();

    const supabase = getDb();

    // Update user name and experience level if provided
    if (fullName !== undefined || experienceLevel !== undefined) {
      const updateFields: Record<string, string | null> = {
        updated_at: new Date().toISOString(),
      };
      if (fullName !== undefined) {
        updateFields.full_name = fullName.trim();
      }
      if (experienceLevel !== undefined) {
        updateFields.experience_level = experienceLevel;
      }

      await supabase
        .from('users')
        .update(updateFields)
        .eq('id', user.id);
    }

    // Update interests if provided
    if (interests !== undefined && Array.isArray(interests)) {
      await supabase
        .from('user_interests')
        .delete()
        .eq('user_id', user.id);

      if (interests.length > 0) {
        const rows = interests.map((tag: string) => ({
          id: generateId(),
          user_id: user.id,
          tag,
        }));
        await supabase.from('user_interests').insert(rows);
      }
    }

    // Return updated user
    const updatedUser = await getAuthUser();
    return NextResponse.json({ user: updatedUser });
  } catch (error) {
    console.error('Profile update error:', error);
    return NextResponse.json(
      { error: 'Something went wrong' },
      { status: 500 }
    );
  }
}
