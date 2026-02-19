'use client';

import Link from 'next/link';
import Image from 'next/image';
import { motion } from 'framer-motion';
import Badge from '@/components/ui/Badge';
import { staggerItemVariants } from '@/components/animations/StaggerChildren';
import type { CourseCard as CourseCardType } from '@/types/course';

const difficultyVariant: Record<string, 'blue' | 'green' | 'amber'> = {
  beginner: 'green',
  intermediate: 'blue',
  advanced: 'amber',
};

interface CourseCardProps {
  course: CourseCardType;
  completion?: number;
}

export default function CourseCard({ course, completion = 0 }: CourseCardProps) {
  return (
    <motion.div variants={staggerItemVariants}>
      <Link href={`/courses/${course.slug}`}>
        <div className="group bg-card-bg rounded-2xl overflow-hidden transition-all duration-300 ease-out hover:shadow-xl hover:-translate-y-1 cursor-pointer h-full flex flex-col border border-card-border/60">
          {/* Thumbnail */}
          <div className="relative w-full aspect-[16/9] overflow-hidden bg-gray-100">
            {course.thumbnail ? (
              <Image
                src={course.thumbnail}
                alt={course.title}
                fill
                className="object-cover transition-transform duration-500 group-hover:scale-105"
                sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 33vw"
              />
            ) : (
              <div className="w-full h-full bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
                <svg className="w-12 h-12 text-accent-blue/30" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" />
                </svg>
              </div>
            )}

            {/* Progress overlay */}
            {completion > 0 && (
              <div className="absolute top-3 right-3">
                <div className="bg-white/90 backdrop-blur-sm rounded-full px-2.5 py-1 text-xs font-bold text-accent-green shadow-sm">
                  {completion}%
                </div>
              </div>
            )}
          </div>

          {/* Content */}
          <div className="p-5 flex flex-col flex-1">
            <h3 className="font-semibold text-foreground text-[15px] leading-snug mb-2 group-hover:text-accent-blue transition-colors line-clamp-2">
              {course.title}
            </h3>

            <p className="text-sm text-text-secondary leading-relaxed mb-4 flex-1 line-clamp-2">
              {course.description}
            </p>

            {/* Footer */}
            <div className="flex items-center gap-2 pt-3 border-t border-card-border/50">
              <Badge variant={difficultyVariant[course.difficulty]} size="sm">
                {course.difficulty}
              </Badge>
              <span className="text-xs text-text-muted">~{course.estimatedHours}h</span>
              {course.podCount > 0 && (
                <span className="text-xs text-text-muted ml-auto">
                  {course.podCount} pod{course.podCount !== 1 ? 's' : ''}
                </span>
              )}
              {(course.totalNotebooks || course.notebookCount || 0) > 0 && (
                <span className="text-xs text-text-muted flex items-center gap-1">
                  <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M17.25 6.75L22.5 12l-5.25 5.25m-10.5 0L1.5 12l5.25-5.25m7.5-3l-4.5 16.5" />
                  </svg>
                  {course.totalNotebooks || course.notebookCount}
                </span>
              )}
            </div>

            {/* Progress bar */}
            {completion > 0 && (
              <div className="mt-3">
                <div className="w-full h-1 bg-gray-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-accent-green rounded-full transition-all duration-500"
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
