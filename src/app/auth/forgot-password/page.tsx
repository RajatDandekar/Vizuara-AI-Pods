'use client';

import { useEffect } from 'react';

const VIZUARA_URL = process.env.NEXT_PUBLIC_VIZUARA_URL || 'https://vizuara.ai';

export default function ForgotPasswordPage() {
  useEffect(() => {
    window.location.href = `${VIZUARA_URL}/auth/forgot-password`;
  }, []);

  return (
    <div className="min-h-[calc(100vh-3.5rem)] flex items-center justify-center">
      <div className="text-center">
        <div className="w-8 h-8 border-2 border-accent-blue border-t-transparent rounded-full animate-spin mx-auto mb-3" />
        <p className="text-sm text-text-muted">Redirecting...</p>
      </div>
    </div>
  );
}
