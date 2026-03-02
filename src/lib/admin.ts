import { getAuthUser } from './auth';
import type { User } from '@/types/auth';

export async function isAdmin(): Promise<boolean> {
  if (process.env.NODE_ENV === 'development') return true;

  const user = await getAuthUser();
  if (!user) return false;

  const adminUids = process.env.ADMIN_UIDS?.split(',').map(s => s.trim()) ?? [];
  return adminUids.includes(user.id);
}

export async function getAdminUser(): Promise<User | null> {
  // In development, return a stub admin user (login flow requires vizuara.ai)
  if (process.env.NODE_ENV === 'development') {
    const user = await getAuthUser();
    if (user) return user;

    // No session cookie in dev — return stub
    return {
      id: process.env.ADMIN_UIDS?.split(',')[0]?.trim() || 'dev-admin',
      fullName: 'Dev Admin',
      email: 'admin@dev.local',
      experienceLevel: null,
      onboardingComplete: true,
      interests: [],
    };
  }

  const user = await getAuthUser();
  if (!user) return null;

  const adminUids = process.env.ADMIN_UIDS?.split(',').map(s => s.trim()) ?? [];
  if (!adminUids.includes(user.id)) return null;

  return user;
}
