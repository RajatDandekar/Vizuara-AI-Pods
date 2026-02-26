'use client';

import { useEffect } from 'react';
import { usePathname } from 'next/navigation';
import { useAuth } from '@/context/AuthContext';
import { useSubscription } from '@/context/SubscriptionContext';

const VIZUARA_URL = process.env.NEXT_PUBLIC_VIZUARA_URL || 'https://vizuara.ai';

// Pods that are free for all logged-in users (no subscription required)
const FREE_PODS = new Set([
  'intro-ddpm',
  'basics-of-rl',
  'vision-encoders',
]);

interface SubscriptionGateProps {
  children: React.ReactNode;
  podSlug?: string;
}

export default function SubscriptionGate({ children, podSlug }: SubscriptionGateProps) {
  const { user, loading: authLoading } = useAuth();
  const { enrolled, loading: subLoading } = useSubscription();
  const pathname = usePathname();

  const isFree = podSlug ? FREE_PODS.has(podSlug) : false;
  const loading = authLoading || (isFree ? false : subLoading);
  const hasAccess = isFree ? !!user : !!user && enrolled;

  useEffect(() => {
    if (loading) return;
    if (!user) {
      window.location.href = `/api/auth/redirect?returnTo=${encodeURIComponent(pathname)}`;
    } else if (!isFree && !enrolled) {
      window.location.href = `${VIZUARA_URL}/courses/ai-pods`;
    }
  }, [user, enrolled, loading, pathname, isFree]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-accent-blue border-t-transparent rounded-full animate-spin mx-auto mb-3" />
          <p className="text-sm text-text-muted">Loading...</p>
        </div>
      </div>
    );
  }

  if (!hasAccess) return null;

  return <>{children}</>;
}
