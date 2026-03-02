import { redirect } from 'next/navigation';
import { getAdminUser } from '@/lib/admin';
import { CreatorProvider } from '@/context/CreatorContext';

export default async function CreatorLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  // In development, skip auth check (login flow requires vizuara.ai redirect)
  const isDev = process.env.NODE_ENV === 'development';
  if (!isDev) {
    const admin = await getAdminUser();
    if (!admin) redirect('/');
  }

  return <CreatorProvider>{children}</CreatorProvider>;
}
