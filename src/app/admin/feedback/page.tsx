import { redirect } from 'next/navigation';
import { getAdminUser } from '@/lib/admin';
import AdminFeedbackDashboard from './AdminFeedbackDashboard';

export default async function AdminFeedbackPage() {
  const admin = await getAdminUser();
  if (!admin) redirect('/');

  return <AdminFeedbackDashboard />;
}
