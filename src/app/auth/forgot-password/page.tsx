'use client';

import Link from 'next/link';
import Image from 'next/image';
import FadeIn from '@/components/animations/FadeIn';
import Button from '@/components/ui/Button';

export default function ForgotPasswordPage() {
  return (
    <div className="min-h-[calc(100vh-3.5rem)] flex items-center justify-center px-4 py-12 bg-gray-50">
      <FadeIn className="w-full max-w-md">
        <div className="bg-card-bg border border-card-border rounded-2xl p-8 shadow-sm text-center">
          <Link href="/" className="inline-flex items-center gap-2 mb-6">
            <Image src="/vizuara-logo.png" alt="Vizuara AI Pods" width={32} height={32} className="rounded-md" />
            <span className="font-bold text-lg text-foreground">Vizuara AI Pods</span>
          </Link>

          <div className="w-16 h-16 bg-blue-50 rounded-2xl flex items-center justify-center mx-auto mb-6">
            <svg className="w-8 h-8 text-accent-blue" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M21.75 6.75v10.5a2.25 2.25 0 01-2.25 2.25h-15a2.25 2.25 0 01-2.25-2.25V6.75m19.5 0A2.25 2.25 0 0019.5 4.5h-15a2.25 2.25 0 00-2.25 2.25m19.5 0v.243a2.25 2.25 0 01-1.07 1.916l-7.5 4.615a2.25 2.25 0 01-2.36 0L3.32 8.91a2.25 2.25 0 01-1.07-1.916V6.75" />
            </svg>
          </div>

          <h1 className="text-2xl font-bold text-foreground mb-2">Reset your password</h1>
          <p className="text-sm text-text-secondary mb-8 leading-relaxed">
            This feature is coming soon. For now, please contact us at{' '}
            <span className="text-accent-blue font-medium">support@vizuara.ai</span>{' '}
            to reset your password.
          </p>

          <Link href="/auth/login">
            <Button variant="secondary" size="md">
              Back to Login
            </Button>
          </Link>
        </div>
      </FadeIn>
    </div>
  );
}
