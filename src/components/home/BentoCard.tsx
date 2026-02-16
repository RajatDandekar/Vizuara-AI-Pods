'use client';

import Link from 'next/link';
import Image from 'next/image';
import { motion } from 'framer-motion';
import type { CourseCard } from '@/types/course';

type BentoSize = 'standard' | 'wide' | 'featured';

const difficultyColor: Record<string, string> = {
  beginner: 'bg-emerald-50 text-emerald-700 border-emerald-200',
  intermediate: 'bg-blue-50 text-blue-700 border-blue-200',
  advanced: 'bg-amber-50 text-amber-700 border-amber-200',
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
              <div className="absolute inset-0 bg-gradient-to-t from-white via-white/30 to-transparent" />

              {/* Progress badge */}
              {completion > 0 && (
                <div className="absolute top-3 right-3">
                  <div className="bg-emerald-50 backdrop-blur-sm border border-emerald-200 rounded-full px-2.5 py-1 text-xs font-bold text-emerald-700">
                    {completion}%
                  </div>
                </div>
              )}

              {/* Upcoming badge */}
              {course.status === 'upcoming' && (
                <div className="absolute top-3 left-3">
                  <div className="bg-amber-50 backdrop-blur-sm border border-amber-200 rounded-full px-2.5 py-1 text-xs font-medium text-amber-700">
                    Coming Soon
                  </div>
                </div>
              )}
            </div>
          )}

          {/* No thumbnail fallback */}
          {!course.thumbnail && (
            <div className={`relative w-full overflow-hidden ${isFeatured ? 'h-40' : 'h-24'} bg-gradient-to-br from-blue-50 to-indigo-50 flex items-center justify-center`}>
              <svg className="w-10 h-10 text-blue-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" />
              </svg>
              {course.status === 'upcoming' && (
                <div className="absolute top-3 left-3">
                  <div className="bg-amber-50 backdrop-blur-sm border border-amber-200 rounded-full px-2.5 py-1 text-xs font-medium text-amber-700">
                    Coming Soon
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Content */}
          <div className={`p-4 ${isFeatured ? 'sm:p-5' : ''} flex flex-col flex-1`}>
            <h3 className={`font-semibold text-slate-900 ${isFeatured ? 'text-lg' : 'text-sm'} leading-snug mb-1.5 group-hover:text-blue-600 transition-colors line-clamp-2`}>
              {course.title}
            </h3>

            <p className={`text-slate-500 leading-relaxed mb-3 flex-1 ${isFeatured ? 'text-sm line-clamp-3' : 'text-xs line-clamp-2'}`}>
              {course.description}
            </p>

            {/* Meta */}
            <div className="flex items-center gap-2 pt-2 border-t border-slate-100">
              <span className={`text-[11px] font-medium px-2 py-0.5 rounded-full border ${difficultyColor[course.difficulty]}`}>
                {course.difficulty}
              </span>
              <span className="text-[11px] text-slate-400">~{course.estimatedHours}h</span>
              {course.notebookCount > 0 && (
                <span className="text-[11px] text-slate-400 flex items-center gap-1 ml-auto">
                  <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M17.25 6.75L22.5 12l-5.25 5.25m-10.5 0L1.5 12l5.25-5.25m7.5-3l-4.5 16.5" />
                  </svg>
                  {course.notebookCount}
                </span>
              )}
            </div>

            {/* Progress bar */}
            {completion > 0 && (
              <div className="mt-2.5">
                <div className="w-full h-1 bg-slate-100 rounded-full overflow-hidden">
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
