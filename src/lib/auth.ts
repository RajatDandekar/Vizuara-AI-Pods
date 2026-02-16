import { SignJWT, jwtVerify } from 'jose';
import { hash, compare } from 'bcryptjs';
import { cookies } from 'next/headers';
import { getDb } from './db';
import type { User } from '@/types/auth';

const COOKIE_NAME = 'vizflix_token';
const JWT_SECRET = new TextEncoder().encode(
  process.env.JWT_SECRET || 'vizflix-dev-secret-change-in-production'
);

// Password helpers
export async function hashPassword(password: string): Promise<string> {
  return hash(password, 12);
}

export async function verifyPassword(password: string, passwordHash: string): Promise<boolean> {
  return compare(password, passwordHash);
}

// JWT helpers
export async function signToken(userId: string, rememberMe = false): Promise<string> {
  const expiresIn = rememberMe ? '7d' : '24h';
  return new SignJWT({ userId })
    .setProtectedHeader({ alg: 'HS256' })
    .setExpirationTime(expiresIn)
    .setIssuedAt()
    .sign(JWT_SECRET);
}

export async function verifyToken(token: string): Promise<{ userId: string } | null> {
  try {
    const { payload } = await jwtVerify(token, JWT_SECRET);
    return { userId: payload.userId as string };
  } catch {
    return null;
  }
}

// Cookie helpers
export async function setAuthCookie(token: string, rememberMe = false) {
  const cookieStore = await cookies();
  cookieStore.set(COOKIE_NAME, token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    path: '/',
    maxAge: rememberMe ? 7 * 24 * 60 * 60 : 24 * 60 * 60,
  });
}

export async function clearAuthCookie() {
  const cookieStore = await cookies();
  cookieStore.set(COOKIE_NAME, '', {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    path: '/',
    maxAge: 0,
  });
}

export async function getAuthToken(): Promise<string | null> {
  const cookieStore = await cookies();
  return cookieStore.get(COOKIE_NAME)?.value ?? null;
}

// Get authenticated user from cookie
export async function getAuthUser(): Promise<User | null> {
  const token = await getAuthToken();
  if (!token) return null;

  const payload = await verifyToken(token);
  if (!payload) return null;

  const supabase = getDb();

  const { data: row, error: userError } = await supabase
    .from('users')
    .select('id, full_name, email, experience_level, onboarding_complete')
    .eq('id', payload.userId)
    .single();

  if (userError || !row) return null;

  const { data: interests } = await supabase
    .from('user_interests')
    .select('tag')
    .eq('user_id', row.id);

  return {
    id: row.id,
    fullName: row.full_name,
    email: row.email,
    experienceLevel: row.experience_level as User['experienceLevel'],
    onboardingComplete: row.onboarding_complete === true,
    interests: (interests ?? []).map((i: { tag: string }) => i.tag),
  };
}

// Generate a simple unique ID
export function generateId(): string {
  return crypto.randomUUID();
}
