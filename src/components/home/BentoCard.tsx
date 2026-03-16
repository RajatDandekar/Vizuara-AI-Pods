'use client';

import Link from 'next/link';
import Image from 'next/image';
import { motion } from 'framer-motion';
import type { CourseCard } from '@/types/course';

type BentoSize = 'standard' | 'wide' | 'featured';

const difficultyColor: Record<string, string> = {
  beginner: 'difficulty-beginner border',
  intermediate: 'difficulty-intermediate border',
  advanced: 'difficulty-advanced border',
};

const sizeClasses: Record<BentoSize, string> = {
  standard: 'col-span-1 row-span-1',
  wide: 'sm:col-span-2 row-span-1',
  featured: 'sm:col-span-2 sm:row-span-2',
};

interface BentoCardProps {
  course: CourseCard;
  size?: BentoSize;
  completion?: number;
}

export default function BentoCard({ course, size = 'standard', completion = 0 }: BentoCardProps) {
  const isFeatured = size === 'featured';

  return (
    <motion.div
      className={sizeClasses[size]}
      whileHover={{ scale: 1.02 }}
      transition={{ duration: 0.2 }}
    >
      <Link href={course.status === 'upcoming' ? '#' : `/courses/${course.slug}`}>
        <div className="glass-card rounded-2xl overflow-hidden h-full flex flex-col cursor-pointer group relative">
          {/* Thumbnail */}
          {course.thumbnail && (
            <div className={`relative w-full overflow-hidden ${isFeatured ? 'aspect-[16/10] sm:flex-1 sm:min-h-0' : 'aspect-[16/9]'}`}>
              <Image
                src={course.thumbnail}
                alt={course.title}
                fill
                className="object-cover transition-transform duration-500 group-hover:scale-105"
                sizes={isFeatured ? '(max-width: 640px) 100vw, 50vw' : '(max-width: 640px) 100vw, 25vw'}
              />
              <div className="absolute inset-0 bg-gradient-to-t from-gradient-overlay-from via-gradient-overlay-from/30 to-transparent" />

              {/* Progress badge */}
              {completion > 0 && (
                <div className="absolute top-3 right-3">
                  <div className="bg-accent-green-light backdrop-blur-sm border border-accent-green/30 rounded-full px-2.5 py-1 text-xs font-bold text-accent-green">
                    {completion}%
                  </div>
                </div>
              )}

              {/* Upcoming badge */}
              {course.status === 'upcoming' && (
                <div className="absolute top-3 left-3">
                  <div className="bg-accent-amber-light backdrop-blur-sm border border-accent-amber/30 rounded-full px-2.5 py-1 text-xs font-medium text-accent-amber">
                    Coming Soon
                  </div>
                </div>
              )}
            </div>
          )}

          {/* No thumbnail fallback */}
          {!course.thumbnail && (
            <div className={`relative w-full overflow-hidden ${isFeatured ? 'h-40' : 'h-24'} bg-gradient-to-br from-gradient-from to-gradient-to flex items-center justify-center`}>
              <svg className="w-10 h-10 text-accent-blue/40" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" />
              </svg>
              {course.status === 'upcoming' && (
                <div className="absolute top-3 left-3">
                  <div className="bg-accent-amber-light backdrop-blur-sm border border-accent-amber/30 rounded-full px-2.5 py-1 text-xs font-medium text-accent-amber">
                    Coming Soon
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Content */}
          <div className={`p-4 ${isFeatured ? 'sm:p-5' : ''} flex flex-col flex-1`}>
            <h3 className={`font-semibold text-foreground ${isFeatured ? 'text-lg' : 'text-sm'} leading-snug mb-1.5 group-hover:text-accent-blue transition-colors line-clamp-2`}>
              {course.title}
            </h3>

            <p className={`text-text-muted leading-relaxed mb-3 flex-1 ${isFeatured ? 'text-sm line-clamp-3' : 'text-xs line-clamp-2'}`}>
              {course.description}
            </p>

            {/* Meta */}
            <div className="flex items-center gap-2 pt-2 border-t border-card-border/30">
              <span className={`text-[11px] font-medium px-2 py-0.5 rounded-full border ${difficultyColor[course.difficulty]}`}>
                {course.difficulty}
              </span>
              <span className="text-[11px] text-text-muted">~{course.estimatedHours}h</span>
              {(course.totalNotebooks || course.notebookCount || 0) > 0 && (
                <span className="text-[11px] text-text-muted flex items-center gap-1 ml-auto">
                  <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M17.25 6.75L22.5 12l-5.25 5.25m-10.5 0L1.5 12l5.25-5.25m7.5-3l-4.5 16.5" />
                  </svg>
                  {course.totalNotebooks || course.notebookCount}
                </span>
              )}
            </div>

            {/* Progress bar */}
            {completion > 0 && (
              <div className="mt-2.5">
                <div className="w-full h-1 bg-progress-track rounded-full overflow-hidden">
                  <div
                    className="h-full bg-emerald-500 rounded-full transition-all duration-500"
                    style={{ width: `${completion}%` }}
                  />
                </div>
              </div>
            )}
          </div>
        </div>
      </Link>
    </motion.div>
  );
}
