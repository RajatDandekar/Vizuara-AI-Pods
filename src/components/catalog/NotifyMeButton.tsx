'use client';

import { useState, useEffect } from 'react';
import { useAuth } from '@/context/AuthContext';
import Button from '@/components/ui/Button';

interface NotifyMeButtonProps {
  courseSlug: string;
  size?: 'sm' | 'md';
}

export default function NotifyMeButton({ courseSlug, size = 'sm' }: NotifyMeButtonProps) {
  const { user } = useAuth();
  const [subscribed, setSubscribed] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!user) return;
    fetch(`/api/notifications/status?courseSlug=${courseSlug}`)
      .then((res) => res.json())
      .then((data) => setSubscribed(data.subscribed))
      .catch(() => {});
  }, [user, courseSlug]);

  async function handleToggle(e: React.MouseEvent) {
    e.preventDefault();
    e.stopPropagation();

    if (!user) return;

    setLoading(true);
    try {
      const method = subscribed ? 'DELETE' : 'POST';
      const res = await fetch('/api/notifications/subscribe', {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ courseSlug }),
      });
      const data = await res.json();
      setSubscribed(data.subscribed);
    } catch {
      // ignore
    }
    setLoading(false);
  }

  if (!user) {
    return (
      <Button variant="secondary" size={size} disabled>
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M14.857 17.082a23.848 23.848 0 005.454-1.31A8.967 8.967 0 0118 9.75v-.7V9A6 6 0 006 9v.75a8.967 8.967 0 01-2.312 6.022c1.733.64 3.56 1.085 5.455 1.31m5.714 0a24.255 24.255 0 01-5.714 0m5.714 0a3 3 0 11-5.714 0" />
        </svg>
        Sign in to get notified
      </Button>
    );
  }

  return (
    <Button
      variant={subscribed ? 'primary' : 'secondary'}
      size={size}
      onClick={handleToggle}
      isLoading={loading}
    >
      <svg className="w-4 h-4" fill={subscribed ? 'currentColor' : 'none'} viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M14.857 17.082a23.848 23.848 0 005.454-1.31A8.967 8.967 0 0118 9.75v-.7V9A6 6 0 006 9v.75a8.967 8.967 0 01-2.312 6.022c1.733.64 3.56 1.085 5.455 1.31m5.714 0a24.255 24.255 0 01-5.714 0m5.714 0a3 3 0 11-5.714 0" />
      </svg>
      {subscribed ? "You'll be notified" : 'Notify Me'}
      {subscribed && <span className="ml-0.5">&#10003;</span>}
    </Button>
  );
}
