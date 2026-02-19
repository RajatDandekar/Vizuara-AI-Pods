'use client';

import { useEffect } from 'react';

export default function LoginPage() {
  useEffect(() => {
    // Redirect to the server-side redirect handler which sets
    // a return cookie and sends the user to vizuara.ai login
    window.location.href = '/api/auth/redirect?returnTo=/';
  }, []);

  return (
    <div className="min-h-[calc(100vh-3.5rem)] flex items-center justify-center">
      <div className="text-center">
        <div className="w-8 h-8 border-2 border-accent-blue border-t-transparent rounded-full animate-spin mx-auto mb-3" />
        <p className="text-sm text-text-muted">Redirecting to login...</p>
      </div>
    </div>
  );
}
