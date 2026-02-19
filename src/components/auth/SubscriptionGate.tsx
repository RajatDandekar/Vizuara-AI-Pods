'use client';

import { useEffect } from 'react';
import { usePathname } from 'next/navigation';
import { useAuth } from '@/context/AuthContext';
import { useSubscription } from '@/context/SubscriptionContext';

const VIZUARA_URL = process.env.NEXT_PUBLIC_VIZUARA_URL || 'https://vizuara.ai';

interface SubscriptionGateProps {
  children: React.ReactNode;
}

export default function SubscriptionGate({ children }: SubscriptionGateProps) {
  const { user, loading: authLoading } = useAuth();
  const { enrolled, loading: subLoading } = useSubscription();
  const pathname = usePathname();

  const loading = authLoading || subLoading;

  useEffect(() => {
    if (loading) return;
    if (!user) {
      // Server-side redirect sets return cookie, then sends to vizuara.ai login
      window.location.href = `/api/auth/redirect?returnTo=${encodeURIComponent(pathname)}`;
    } else if (!enrolled) {
      window.location.href = `${VIZUARA_URL}/pricing`;
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

  if (!user || !enrolled) return null;

  return <>{children}</>;
}
