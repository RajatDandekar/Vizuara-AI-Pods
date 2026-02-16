'use client';

import Link from 'next/link';
import Image from 'next/image';
import { motion } from 'framer-motion';
import Badge from '@/components/ui/Badge';
import NotifyMeButton from '@/components/catalog/NotifyMeButton';
import type { CourseCard } from '@/types/course';

const difficultyVariant: Record<string, 'blue' | 'green' | 'amber'> = {
  beginner: 'green',
  intermediate: 'blue',
  advanced: 'amber',
};

interface CourseRowCardProps {
  course: CourseCard;
  completion?: number;
}

export default function CourseRowCard({ course, completion = 0 }: CourseRowCardProps) {
  const isUpcoming = course.status === 'upcoming';

  return (
    <motion.div
      className="flex-shrink-0 w-[260px] sm:w-[280px]"
      whileHover={{ scale: 1.03 }}
      transition={{ duration: 0.2, ease: [0.25, 0.1, 0.25, 1] }}
    >
      <Link href={`/courses/${course.slug}`}>
        <div className="group bg-card-bg rounded-xl overflow-hidden transition-all duration-300 hover:shadow-lg cursor-pointer h-full flex flex-col border border-card-border/60">
          {/* Thumbnail */}
          <div className="relative w-full aspect-[16/9] overflow-hidden bg-gray-100">
            {course.thumbnail ? (
              <Image
                src={course.thumbnail}
                alt={course.title}
                fill
                className="object-cover transition-transform duration-500 group-hover:scale-105"
                sizes="280px"
              />
            ) : (
              <div className="w-full h-full bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
                <svg className="w-10 h-10 text-accent-blue/30" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" />
                </svg>
              </div>
            )}

            {/* Coming Soon badge */}
            {isUpcoming && (
              <div className="absolute top-2 left-2">
                <span className="bg-amber-500 text-white text-xs font-bold px-2.5 py-1 rounded-full shadow-sm">
                  Coming Soon
                </span>
              </div>
            )}

            {/* Completion badge */}
            {completion > 0 && !isUpcoming && (
              <div className="absolute top-2 right-2">
                <div className="bg-white/90 backdrop-blur-sm rounded-full px-2 py-0.5 text-xs font-bold text-accent-green shadow-sm">
                  {completion}%
                </div>
              </div>
            )}
          </div>

          {/* Content */}
          <div className="p-4 flex flex-col flex-1">
            <h3 className="font-semibold text-foreground text-sm leading-snug mb-1.5 group-hover:text-accent-blue transition-colors line-clamp-2">
              {course.title}
            </h3>
            <p className="text-xs text-text-secondary leading-relaxed mb-3 flex-1 line-clamp-2">
              {course.description}
            </p>

            <div className="flex items-center gap-2">
              <Badge variant={difficultyVariant[course.difficulty]} size="sm">
                {course.difficulty}
              </Badge>
              {isUpcoming && course.expectedLaunchDate ? (
                <span className="text-xs text-text-muted">
                  {new Date(course.expectedLaunchDate).toLocaleDateString('en-US', { month: 'short', year: 'numeric' })}
                </span>
              ) : (
                <span className="text-xs text-text-muted">~{course.estimatedHours}h</span>
              )}
            </div>

            {/* Progress bar */}
            {completion > 0 && !isUpcoming && (
              <div className="mt-2">
                <div className="w-full h-1 bg-gray-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-accent-green rounded-full transition-all duration-500"
                    style={{ width: `${completion}%` }}
                  />
                </div>
              </div>
            )}

            {/* Notify Me for upcoming */}
            {isUpcoming && (
              <div className="mt-2">
                <NotifyMeButton courseSlug={course.slug} size="sm" />
              </div>
            )}
          </div>
        </div>
      </Link>
    </motion.div>
  );
}
