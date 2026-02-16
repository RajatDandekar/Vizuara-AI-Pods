import { NextRequest, NextResponse } from 'next/server';
import { getDb } from '@/lib/db';
import { verifyPassword, signToken, setAuthCookie } from '@/lib/auth';

export const dynamic = 'force-dynamic';

export async function POST(request: NextRequest) {
  try {
    const { email, password, rememberMe = false } = await request.json();

    if (!email || !password) {
      return NextResponse.json(
        { error: 'Email and password are required' },
        { status: 400 }
      );
    }

    const supabase = getDb();

    const { data: row, error } = await supabase
      .from('users')
      .select('id, full_name, email, password_hash, experience_level, onboarding_complete')
      .eq('email', email.toLowerCase().trim())
      .maybeSingle();

    if (error || !row) {
      return NextResponse.json(
        { error: 'Invalid email or password' },
        { status: 401 }
      );
    }

    const valid = await verifyPassword(password, row.password_hash);
    if (!valid) {
      return NextResponse.json(
        { error: 'Invalid email or password' },
        { status: 401 }
      );
    }

    // Get user interests
    const { data: interests } = await supabase
      .from('user_interests')
      .select('tag')
      .eq('user_id', row.id);

    // Sign token and set cookie
    const token = await signToken(row.id, rememberMe);
    await setAuthCookie(token, rememberMe);

    return NextResponse.json({
      user: {
        id: row.id,
        fullName: row.full_name,
        email: row.email,
        experienceLevel: row.experience_level,
        onboardingComplete: row.onboarding_complete === true,
        interests: (interests ?? []).map((i: { tag: string }) => i.tag),
      },
    });
  } catch (error) {
    console.error('Login error:', error);
    return NextResponse.json(
      { error: 'Something went wrong. Please try again.' },
      { status: 500 }
    );
  }
}
