import { getAuthUser } from './auth';

export async function isAdmin(): Promise<boolean> {
  const user = await getAuthUser();
  if (!user) return false;

  const adminUids = process.env.ADMIN_UIDS?.split(',').map(s => s.trim()) ?? [];
  return adminUids.includes(user.id);
}

export async function getAdminUser() {
  const user = await getAuthUser();
  if (!user) return null;

  const adminUids = process.env.ADMIN_UIDS?.split(',').map(s => s.trim()) ?? [];
  if (!adminUids.includes(user.id)) return null;

  return user;
}
