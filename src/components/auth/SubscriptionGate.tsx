'use client';

import { useEffect } from 'react';
import { usePathname } from 'next/navigation';
import { useAuth } from '@/context/AuthContext';
import { useSubscription } from '@/context/SubscriptionContext';
import { FREE_POD_SLUGS } from '@/lib/constants';

const VIZUARA_URL = process.env.NEXT_PUBLIC_VIZUARA_URL || 'https://vizuara.ai';

interface SubscriptionGateProps {
  children: React.ReactNode;
  podSlug?: string;
}

export default function SubscriptionGate({ children, podSlug }: SubscriptionGateProps) {
  const { user, loading: authLoading } = useAuth();
  const { enrolled, loading: subLoading } = useSubscription();
  const pathname = usePathname();

  const isFree = podSlug ? FREE_POD_SLUGS.has(podSlug) : false;

  // Free pods: no loading gate at all — allow immediate access
  if (isFree) return <>{children}</>;

  const loading = authLoading || subLoading;
  const hasAccess = !!user && enrolled;

  useEffect(() => {
    if (loading) return;
    if (!user) {
      window.location.href = `/api/auth/redirect?returnTo=${encodeURIComponent(pathname)}`;
    } else if (!enrolled) {
      window.location.href = `${VIZUARA_URL}/courses/ai-pods`;
    }
  }, [user, enrolled, loading, pathname]);

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
