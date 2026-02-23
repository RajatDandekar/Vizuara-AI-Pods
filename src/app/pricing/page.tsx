'use client';

import { useAuth } from '@/context/AuthContext';
import { useSubscription } from '@/context/SubscriptionContext';
import Button from '@/components/ui/Button';
import FadeIn from '@/components/animations/FadeIn';

const VIZUARA_URL = process.env.NEXT_PUBLIC_VIZUARA_URL || 'https://vizuara.ai';
const PODS_CALLBACK_URL = process.env.NEXT_PUBLIC_PODS_CALLBACK_URL || 'https://pods.vizuara.ai/api/auth/session';

const FEATURES = [
  'All courses — current and future',
  'Hands-on Colab notebooks',
  'Real-world case studies',
  'Practice problems with AI assistant',
  'Course completion certificates',
  'New courses added every month',
];

export default function PricingPage() {
  const { user } = useAuth();
  const { enrolled, loading } = useSubscription();

  return (
    <div className="max-w-lg mx-auto px-4 sm:px-6 py-16">
      <FadeIn>
        <div className="text-center mb-10">
          <h1 className="text-3xl font-bold text-foreground mb-3">
            Vizuara AI Pods
          </h1>
          <p className="text-text-secondary">
            Master AI from first principles with hands-on courses
          </p>
        </div>

        <div className="bg-card-bg border border-card-border rounded-2xl p-8">
          {/* Price */}
          <div className="text-center mb-8">
            <div className="flex items-baseline justify-center gap-1">
              <span className="text-4xl font-bold text-foreground">₹849</span>
              <span className="text-text-muted">/month</span>
            </div>
            <p className="text-sm text-text-muted mt-1">Cancel anytime</p>
          </div>

          {/* Features */}
          <ul className="space-y-3 mb-8">
            {FEATURES.map((feature) => (
              <li key={feature} className="flex items-start gap-3">
                <svg
                  className="w-5 h-5 text-emerald-500 mt-0.5 shrink-0"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M4.5 12.75l6 6 9-13.5"
                  />
                </svg>
                <span className="text-sm text-text-secondary">{feature}</span>
              </li>
            ))}
          </ul>

          {/* CTA */}
          {loading ? (
            <div className="flex justify-center">
              <div className="w-6 h-6 border-2 border-accent-blue border-t-transparent rounded-full animate-spin" />
            </div>
          ) : enrolled ? (
            <div className="text-center">
              <div className="inline-flex items-center gap-2 px-4 py-2 bg-emerald-50 text-emerald-700 rounded-xl text-sm font-medium">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                You&apos;re enrolled
              </div>
              <p className="text-xs text-text-muted mt-2">
                Manage your subscription on{' '}
                <a href={`${VIZUARA_URL}/courses/ai-pods`} className="text-accent-blue hover:underline">
                  vizuara.ai
                </a>
              </p>
            </div>
          ) : user ? (
            <a href={`${VIZUARA_URL}/courses/ai-pods`}>
              <Button size="lg" className="w-full">
                Subscribe Now
              </Button>
            </a>
          ) : (
            <a href={`${VIZUARA_URL}/auth/signup?redirect=${encodeURIComponent(PODS_CALLBACK_URL)}`}>
              <Button size="lg" className="w-full">
                Sign Up to Subscribe
              </Button>
            </a>
          )}
        </div>
      </FadeIn>
    </div>
  );
}
