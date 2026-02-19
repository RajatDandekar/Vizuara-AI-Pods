import { NextRequest, NextResponse } from 'next/server';
import { getAdminAuth } from '@/lib/firebase-admin';
import { getDb } from '@/lib/db';
import { createSessionCookie } from '@/lib/auth';

export const dynamic = 'force-dynamic';

const VIZUARA_URL = process.env.NEXT_PUBLIC_VIZUARA_URL || 'https://vizuara.ai';
const COOKIE_NAME = 'vizuara_session';
const SESSION_EXPIRY = 14 * 24 * 60 * 60 * 1000; // 14 days
const RETURN_COOKIE = 'pods_return_to';

function getCookieDomain(): string | undefined {
  if (process.env.NODE_ENV !== 'production') return undefined;
  return process.env.COOKIE_DOMAIN || undefined;
}

/**
 * GET /api/auth/session?token={idToken}
 *
 * Callback handler: vizuara.ai redirects here after login with a Firebase ID token.
 * Verifies the token, creates a session cookie, upserts the Supabase user
 * (for preferences), then redirects to the stored return path or /.
 *
 * IMPORTANT: Cookies are set directly on the NextResponse.redirect() object
 * to ensure they survive the redirect. Using cookies().set() + NextResponse.redirect()
 * separately causes the Set-Cookie header to be dropped.
 */
export async function GET(request: NextRequest) {
  const token = request.nextUrl.searchParams.get('token');

  if (!token) {
    return NextResponse.redirect(new URL('/auth/login', request.url));
  }

  try {
    // Verify the Firebase ID token
    const adminAuth = getAdminAuth();
    const decoded = await adminAuth.verifyIdToken(token);

    // Create session cookie
    const sessionCookie = await createSessionCookie(token);

    // Upsert user in Supabase (for preferences storage)
    const supabase = getDb();

    const { data: existing } = await supabase
      .from('users')
      .select('id')
      .eq('id', decoded.uid)
      .maybeSingle();

    if (!existing) {
      const email = decoded.email || '';
      const name = decoded.name || email.split('@')[0];

      // Check for legacy user with same email (migration path)
      const { data: legacyUser } = await supabase
        .from('users')
        .select('id')
        .eq('email', email.toLowerCase())
        .maybeSingle();

      if (legacyUser) {
        // Migrate: move foreign keys to Firebase UID
        await supabase
          .from('user_interests')
          .update({ user_id: decoded.uid })
          .eq('user_id', legacyUser.id);

        const { data: oldUser } = await supabase
          .from('users')
          .select('*')
          .eq('id', legacyUser.id)
          .single();

        await supabase.from('users').delete().eq('id', legacyUser.id);
        await supabase.from('users').insert({
          id: decoded.uid,
          full_name: oldUser?.full_name || name,
          email: email.toLowerCase(),
          experience_level: oldUser?.experience_level || null,
          onboarding_complete: oldUser?.onboarding_complete || false,
          created_at: oldUser?.created_at,
          updated_at: new Date().toISOString(),
        });
      } else {
        await supabase.from('users').insert({
          id: decoded.uid,
          full_name: name,
          email: email.toLowerCase(),
        });
      }
    }

    // Read return-to cookie from the incoming request
    const returnTo = request.cookies.get(RETURN_COOKIE)?.value || '/';

    // Validate returnTo is a safe relative path
    const redirectPath = returnTo.startsWith('/') && !returnTo.startsWith('//') ? returnTo : '/';

    // Build redirect response and set cookies directly on it
    const response = NextResponse.redirect(new URL(redirectPath, request.url));

    // Set session cookie on the response
    const cookieDomain = getCookieDomain();
    response.cookies.set(COOKIE_NAME, sessionCookie, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax',
      path: '/',
      maxAge: SESSION_EXPIRY / 1000,
      ...(cookieDomain ? { domain: cookieDomain } : {}),
    });

    // Clear the return-to cookie
    response.cookies.set(RETURN_COOKIE, '', { maxAge: 0, path: '/' });

    return response;
  } catch (error) {
    console.error('Auth callback error:', error);
    return NextResponse.redirect(`${VIZUARA_URL}/auth/login`);
  }
}
