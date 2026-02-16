'use client';

import { useEffect, useRef } from 'react';
import { useCourse } from '@/context/CourseContext';
import { usePaperDiscovery } from '@/hooks/usePaperDiscovery';
import StaggerChildren from '@/components/animations/StaggerChildren';
import FadeIn from '@/components/animations/FadeIn';
import PaperDiscoveryCard from './PaperDiscoveryCard';
import { PaperCardSkeleton } from '@/components/ui/Skeleton';

interface PaperTimelineProps {
  concept: string;
}

export default function PaperTimeline({ concept }: PaperTimelineProps) {
  const { state } = useCourse();
  const { discoverPapers } = usePaperDiscovery();
  const hasStartedRef = useRef(false);

  // Discover papers on mount (guard against double-fire in strict mode)
  useEffect(() => {
    if (concept && !hasStartedRef.current) {
      hasStartedRef.current = true;
      discoverPapers(concept);
    }
  }, [concept, discoverPapers]);

  // Loading state
  if (state.status === 'discovering') {
    return (
      <div className="space-y-4">
        <FadeIn>
          <div className="text-center mb-8">
            <div className="inline-flex items-center gap-2.5 px-4 py-2 bg-accent-blue-light text-accent-blue rounded-full text-sm font-medium">
              <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Discovering historically significant papers...
            </div>
          </div>
        </FadeIn>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <PaperCardSkeleton key={i} />
          ))}
        </div>
      </div>
    );
  }

  // Error state
  if (state.status === 'error') {
    return (
      <FadeIn>
        <div className="text-center py-12">
          <div className="inline-flex flex-col items-center gap-4 p-8 bg-red-50 border border-red-200 rounded-2xl max-w-md">
            <svg className="w-10 h-10 text-accent-red" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
            </svg>
            <div>
              <p className="font-semibold text-foreground mb-1">Something went wrong</p>
              <p className="text-sm text-text-secondary">{state.error}</p>
            </div>
            <button
              onClick={() => discoverPapers(concept)}
              className="px-4 py-2 bg-accent-blue text-white rounded-xl text-sm font-medium hover:bg-blue-700 transition-colors cursor-pointer"
            >
              Try Again
            </button>
          </div>
        </div>
      </FadeIn>
    );
  }

  // Ready state — show papers as grid
  if (state.status === 'ready' && state.papers.length > 0) {
    return (
      <div>
        {/* Header */}
        <FadeIn>
          <div className="text-center mb-8">
            <h2 className="text-2xl font-bold text-foreground mb-2">
              {concept}
            </h2>
            <p className="text-text-secondary">
              {state.papers.length} landmark papers. Click any paper to explore in depth.
            </p>
          </div>
        </FadeIn>

        {/* Card grid */}
        <StaggerChildren className="grid grid-cols-1 md:grid-cols-2 gap-5" staggerDelay={0.1}>
          {state.papers.map((paper, index) => (
            <PaperDiscoveryCard
              key={paper.id}
              paper={paper}
              concept={concept}
              index={index}
            />
          ))}
        </StaggerChildren>
      </div>
    );
  }

  return null;
}
