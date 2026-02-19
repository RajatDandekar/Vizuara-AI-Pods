import { NextRequest, NextResponse } from 'next/server';
import { cookies } from 'next/headers';

export const dynamic = 'force-dynamic';

const VIZUARA_URL = process.env.NEXT_PUBLIC_VIZUARA_URL || 'https://vizuara.ai';
const PODS_CALLBACK_URL = process.env.NEXT_PUBLIC_PODS_CALLBACK_URL || 'https://pods.vizuara.ai/api/auth/session';
const RETURN_COOKIE = 'pods_return_to';
const RETURN_COOKIE_TTL = 10 * 60; // 10 minutes

/**
 * GET /api/auth/redirect?returnTo={path}
 *
 * Sets a pods_return_to cookie (so the callback knows where to send the user),
 * then redirects to vizuara.ai/auth/login with a redirect param.
 */
export async function GET(request: NextRequest) {
  const returnTo = request.nextUrl.searchParams.get('returnTo') || '/';

  // Validate returnTo is a safe relative path
  const safePath = returnTo.startsWith('/') && !returnTo.startsWith('//') ? returnTo : '/';

  // Set cookie so callback handler knows where to go after auth
  const cookieStore = await cookies();
  cookieStore.set(RETURN_COOKIE, safePath, {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    path: '/',
    maxAge: RETURN_COOKIE_TTL,
  });

  // Redirect to vizuara.ai login with callback URL
  const loginUrl = `${VIZUARA_URL}/auth/login?redirect=${encodeURIComponent(PODS_CALLBACK_URL)}`;

  return NextResponse.redirect(loginUrl);
}
