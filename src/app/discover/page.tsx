'use client';

import { useSearchParams } from 'next/navigation';
import { Suspense } from 'react';
import PaperTimeline from '@/components/papers/PaperTimeline';
import FadeIn from '@/components/animations/FadeIn';
import Link from 'next/link';
import ConceptInput from '@/components/concept/ConceptInput';

function DiscoverContent() {
  const searchParams = useSearchParams();
  const concept = searchParams.get('concept') || '';

  if (!concept) {
    return (
      <div>
        {/* Dark hero — matches home page style */}
        <div className="relative overflow-hidden bg-gradient-to-br from-slate-900 via-blue-950 to-indigo-900">
          <div
            className="absolute inset-0 opacity-[0.04]"
            style={{
              backgroundImage: 'radial-gradient(circle at 1px 1px, white 1px, transparent 0)',
              backgroundSize: '32px 32px',
            }}
          />
          <div className="absolute top-0 left-1/3 w-72 h-72 bg-purple-500/10 rounded-full blur-3xl" />
          <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl" />

          <div className="relative max-w-5xl mx-auto px-4 sm:px-6 pt-16 pb-20">
            <FadeIn className="text-center">
              <Link
                href="/"
                className="inline-flex items-center gap-1.5 text-sm text-blue-300/60 hover:text-blue-200 transition-colors mb-8"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
                </svg>
                Back to courses
              </Link>

              <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-white tracking-tight mb-4">
                Discover{' '}
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-blue-300">
                  Papers
                </span>
              </h1>

              <p className="text-lg text-blue-100/70 max-w-lg mx-auto leading-relaxed mb-10">
                Enter any AI/ML concept and explore the landmark papers that shaped it — with summaries, architecture deep dives, and runnable notebooks.
              </p>

              <ConceptInput basePath="/discover" />
            </FadeIn>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 py-8">
      <FadeIn>
        <Link
          href="/discover"
          className="inline-flex items-center gap-1.5 text-sm text-text-muted hover:text-foreground transition-colors mb-6"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
          </svg>
          New search
        </Link>
      </FadeIn>

      <PaperTimeline concept={concept} />
    </div>
  );
}

export default function DiscoverPage() {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center py-20">
          <div className="flex items-center gap-2 text-text-muted">
            <svg className="w-5 h-5 animate-spin" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            Loading...
          </div>
        </div>
      }
    >
      <DiscoverContent />
    </Suspense>
  );
}
