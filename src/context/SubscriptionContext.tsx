'use client';

import { createContext, useContext, useState, useEffect, useCallback, type ReactNode } from 'react';
import { useAuth } from './AuthContext';
import type { EnrollmentStatus, EnrollmentStatusResponse } from '@/types/subscription';

interface SubscriptionContextValue {
  enrolled: boolean;
  loading: boolean;
  status: EnrollmentStatus | null;
  refreshSubscription: () => Promise<void>;
}

const SubscriptionContext = createContext<SubscriptionContextValue | null>(null);

export function SubscriptionProvider({ children }: { children: ReactNode }) {
  const { user, loading: authLoading } = useAuth();
  const [enrolled, setEnrolled] = useState(false);
  const [status, setStatus] = useState<EnrollmentStatus | null>(null);
  const [loading, setLoading] = useState(true);

  const refreshSubscription = useCallback(async () => {
    try {
      const res = await fetch('/api/subscription/status');
      if (!res.ok) {
        setEnrolled(false);
        setStatus(null);
        setLoading(false);
        return;
      }
      const data: EnrollmentStatusResponse = await res.json();
      setEnrolled(data.enrolled);
      setStatus(data.status);
    } catch {
      setEnrolled(false);
      setStatus(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (authLoading) return;
    if (!user) {
      setEnrolled(false);
      setStatus(null);
      setLoading(false);
      return;
    }
    refreshSubscription();
  }, [user, authLoading, refreshSubscription]);

  return (
    <SubscriptionContext.Provider
      value={{ enrolled, loading: authLoading || loading, status, refreshSubscription }}
    >
      {children}
    </SubscriptionContext.Provider>
  );
}

export function useSubscription(): SubscriptionContextValue {
  const context = useContext(SubscriptionContext);
  if (!context) {
    throw new Error('useSubscription must be used within a SubscriptionProvider');
  }
  return context;
}
