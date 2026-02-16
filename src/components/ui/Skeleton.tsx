'use client';

interface SkeletonProps {
  className?: string;
  lines?: number;
}

export default function Skeleton({ className = '', lines = 1 }: SkeletonProps) {
  if (lines > 1) {
    return (
      <div className="space-y-2.5">
        {Array.from({ length: lines }).map((_, i) => (
          <div
            key={i}
            className={`animate-pulse bg-gray-200 rounded-lg h-4 ${
              i === lines - 1 ? 'w-3/4' : 'w-full'
            } ${className}`}
          />
        ))}
      </div>
    );
  }

  return (
    <div
      className={`animate-pulse bg-gray-200 rounded-lg ${className}`}
    />
  );
}

export function PaperCardSkeleton() {
  return (
    <div className="bg-card-bg border border-card-border rounded-2xl p-6 space-y-3">
      <div className="flex items-center gap-3">
        <div className="animate-pulse bg-gray-200 rounded-full h-6 w-14" />
        <div className="animate-pulse bg-gray-200 rounded-lg h-5 w-48" />
      </div>
      <div className="animate-pulse bg-gray-200 rounded-lg h-4 w-full" />
      <div className="animate-pulse bg-gray-200 rounded-lg h-4 w-3/4" />
    </div>
  );
}
