'use client';

import Link from 'next/link';
import Button from '@/components/ui/Button';

interface CourseNavProps {
  prevHref?: string;
  prevLabel?: string;
  nextHref?: string;
  nextLabel?: string;
  onNext?: () => void;
}

export default function CourseNav({
  prevHref,
  prevLabel,
  nextHref,
  nextLabel,
  onNext,
}: CourseNavProps) {
  return (
    <div className="flex items-center justify-between pt-8 mt-8 border-t border-card-border">
      <div>
        {prevHref && (
          <Link href={prevHref}>
            <Button variant="secondary" size="md">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
              </svg>
              {prevLabel || 'Previous'}
            </Button>
          </Link>
        )}
      </div>
      <div>
        {nextHref && (
          <Link href={nextHref} onClick={onNext}>
            <Button variant="primary" size="md">
              {nextLabel || 'Next'}
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
              </svg>
            </Button>
          </Link>
        )}
      </div>
    </div>
  );
}
