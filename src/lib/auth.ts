import { cookies } from 'next/headers';
import { getAdminAuth } from './firebase-admin';
import { getAdminFirestore } from './firebase-admin';
import { getDb } from './db';
import type { User } from '@/types/auth';

const COOKIE_NAME = 'vizuara_session';
const SESSION_EXPIRY = 14 * 24 * 60 * 60 * 1000; // 14 days

// In production on *.vizuara.ai, set cookie on the parent domain
// so it's shared across pods.vizuara.ai, vizuara.ai, etc.
function getCookieDomain(): string | undefined {
  if (process.env.NODE_ENV !== 'production') return undefined;
  const host = process.env.COOKIE_DOMAIN;
  return host || undefined; // Set COOKIE_DOMAIN=.vizuara.ai in production
}

// Create a Firebase session cookie from an ID token
export async function createSessionCookie(idToken: string): Promise<string> {
  const adminAuth = getAdminAuth();
  return adminAuth.createSessionCookie(idToken, { expiresIn: SESSION_EXPIRY });
}

// Set the session cookie
export async function setAuthCookie(sessionCookie: string) {
  const cookieStore = await cookies();
  cookieStore.set(COOKIE_NAME, sessionCookie, {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    path: '/',
    maxAge: SESSION_EXPIRY / 1000,
    domain: getCookieDomain(),
  });
}

// Clear the session cookie
export async function clearAuthCookie() {
  const cookieStore = await cookies();
  cookieStore.set(COOKIE_NAME, '', {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    path: '/',
    maxAge: 0,
    domain: getCookieDomain(),
  });
}

// Get the session cookie value
async function getSessionCookie(): Promise<string | null> {
  const cookieStore = await cookies();
  return cookieStore.get(COOKIE_NAME)?.value ?? null;
}

// Get authenticated user from Firebase session cookie
export async function getAuthUser(): Promise<User | null> {
  const session = await getSessionCookie();
  if (!session) return null;

  try {
    const adminAuth = getAdminAuth();
    const decoded = await adminAuth.verifySessionCookie(session, true);

    const supabase = getDb();

    const { data: row, error: userError } = await supabase
      .from('users')
      .select('id, full_name, email, experience_level, onboarding_complete')
      .eq('id', decoded.uid)
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
  } catch {
    return null;
  }
}

// Generate a simple unique ID
export function generateId(): string {
  return crypto.randomUUID();
}

// Check if user is enrolled via Firestore (written by vizuara.ai)
const PODS_ENROLLMENT_ID = process.env.PODS_ENROLLMENT_ID || 'course_20006198';

export async function checkEnrollment(uid: string): Promise<boolean> {
  try {
    const db = getAdminFirestore();
    const docRef = db.collection('Enrollments').doc(`${uid}_${PODS_ENROLLMENT_ID}`);
    const doc = await docRef.get();

    if (!doc.exists) return false;

    const data = doc.data();
    return data?.status === 'ACTIVE' || data?.status === 'COMPLETED';
  } catch {
    return false;
  }
}
