'use client';

import { useEffect, useRef } from 'react';
import { useParams, useSearchParams } from 'next/navigation';
import { useCourse } from '@/context/CourseContext';
import { usePaperDiscovery } from '@/hooks/usePaperDiscovery';
import PaperDetailView from '@/components/papers/PaperDetailView';
import FadeIn from '@/components/animations/FadeIn';
import Link from 'next/link';

export default function PaperDetailPage() {
  const params = useParams();
  const searchParams = useSearchParams();

  const paperId = params.paperId as string;
  const concept = searchParams.get('concept') || '';

  const { state } = useCourse();
  const { discoverPapers } = usePaperDiscovery();
  const hasStartedRef = useRef(false);

  // If context is empty (direct URL visit), auto-discover papers
  useEffect(() => {
    if (concept && state.status === 'idle' && !hasStartedRef.current) {
      hasStartedRef.current = true;
      discoverPapers(concept);
    }
  }, [concept, state.status, discoverPapers]);

  const paper = state.papers.find((p) => p.id === paperId);

  // Loading state
  if (state.status === 'discovering') {
    return (
      <div className="max-w-3xl mx-auto px-4 sm:px-6 py-8">
        <FadeIn>
          <div className="flex flex-col items-center justify-center py-20 gap-4">
            <svg className="w-8 h-8 animate-spin text-accent-blue" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            <p className="text-text-secondary">Discovering papers about <span className="font-medium text-foreground">{concept}</span>...</p>
          </div>
        </FadeIn>
      </div>
    );
  }

  // Error state
  if (state.status === 'error') {
    return (
      <div className="max-w-3xl mx-auto px-4 sm:px-6 py-8">
        <FadeIn>
          <Link
            href="/discover"
            className="inline-flex items-center gap-1.5 text-sm text-text-muted hover:text-foreground transition-colors mb-6"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
            </svg>
            Back to search
          </Link>
          <div className="text-center py-12">
            <p className="text-accent-red font-medium mb-2">Failed to load papers</p>
            <p className="text-sm text-text-secondary mb-4">{state.error}</p>
            <button
              onClick={() => discoverPapers(concept)}
              className="px-4 py-2 bg-accent-blue text-white rounded-xl text-sm font-medium hover:bg-blue-700 transition-colors cursor-pointer"
            >
              Try Again
            </button>
          </div>
        </FadeIn>
      </div>
    );
  }

  // Paper not found
  if (state.status === 'ready' && !paper) {
    return (
      <div className="max-w-3xl mx-auto px-4 sm:px-6 py-8">
        <FadeIn>
          <Link
            href={concept ? `/discover?concept=${encodeURIComponent(concept)}` : '/discover'}
            className="inline-flex items-center gap-1.5 text-sm text-text-muted hover:text-foreground transition-colors mb-6"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
            </svg>
            Back to papers
          </Link>
          <div className="text-center py-12">
            <p className="text-foreground font-medium mb-2">Paper not found</p>
            <p className="text-sm text-text-secondary">The paper you&apos;re looking for doesn&apos;t exist or the session has expired.</p>
          </div>
        </FadeIn>
      </div>
    );
  }

  // No concept provided
  if (!concept) {
    return (
      <div className="max-w-3xl mx-auto px-4 sm:px-6 py-8">
        <FadeIn>
          <div className="text-center py-12">
            <p className="text-foreground font-medium mb-2">No concept specified</p>
            <Link
              href="/discover"
              className="text-sm text-accent-blue hover:underline"
            >
              Go to Discover
            </Link>
          </div>
        </FadeIn>
      </div>
    );
  }

  if (!paper) return null;

  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 py-8">
      <FadeIn>
        <PaperDetailView paper={paper} concept={concept} />
      </FadeIn>
    </div>
  );
}
