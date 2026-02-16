'use client';

import { useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import Image from 'next/image';
import {
  motion,
  useMotionValue,
  useTransform,
  useSpring,
} from 'framer-motion';
import Badge from '@/components/ui/Badge';
import NotifyMeButton from '@/components/catalog/NotifyMeButton';
import type { CourseCard } from '@/types/course';

interface CourseExpandedCardProps {
  course: CourseCard;
  completion?: number;
  onClose: () => void;
  onMouseEnter: () => void;
  onMouseLeave: () => void;
}

const difficultyVariant: Record<string, 'blue' | 'green' | 'amber'> = {
  beginner: 'green',
  intermediate: 'blue',
  advanced: 'amber',
};

const ease = [0.25, 0.1, 0.25, 1] as [number, number, number, number];

export default function CourseExpandedCard({
  course,
  completion = 0,
  onClose,
  onMouseEnter,
  onMouseLeave,
}: CourseExpandedCardProps) {
  const router = useRouter();
  const isUpcoming = course.status === 'upcoming';

  // --- Escape key ---
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    },
    [onClose],
  );

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  // --- Mouse parallax ---
  const mouseX = useMotionValue(0.5);
  const mouseY = useMotionValue(0.5);

  const rawX = useTransform(mouseX, [0, 1], [-12, 12]);
  const rawY = useTransform(mouseY, [0, 1], [-8, 8]);
  const imgX = useSpring(rawX, { stiffness: 100, damping: 20 });
  const imgY = useSpring(rawY, { stiffness: 100, damping: 20 });

  function handleMouseMove(e: React.MouseEvent<HTMLDivElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    mouseX.set((e.clientX - rect.left) / rect.width);
    mouseY.set((e.clientY - rect.top) / rect.height);
  }

  // --- CTA ---
  let ctaLabel = 'Start Learning';
  if (isUpcoming) ctaLabel = 'Coming Soon';
  else if (completion >= 100) ctaLabel = 'Review Course';
  else if (completion > 0) ctaLabel = 'Continue Learning';

  function handleCta(e: React.MouseEvent) {
    e.stopPropagation();
    if (isUpcoming) return;
    onClose();
    router.push(`/courses/${course.slug}`);
  }

  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center p-4 pointer-events-none">
      {/* Card — only the card itself captures pointer events */}
      <motion.div
        className="relative w-full max-w-2xl rounded-2xl overflow-hidden bg-white shadow-2xl flex flex-col max-h-[85vh] pointer-events-auto ring-1 ring-black/5"
        initial={{ opacity: 0, scale: 0.92, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: 10 }}
        transition={{ duration: 0.25, ease }}
        onMouseEnter={onMouseEnter}
        onMouseLeave={onMouseLeave}
      >
        {/* Image area */}
        <div
          className="relative w-full aspect-[16/9] overflow-hidden flex-shrink-0"
          onMouseMove={handleMouseMove}
        >
          {/* Ken Burns outer wrapper */}
          <div className="absolute inset-0 kenburns-active">
            {/* Parallax inner wrapper */}
            <motion.div
              className="relative w-full h-full"
              style={{ x: imgX, y: imgY, scale: 1.15 }}
            >
              {course.thumbnail ? (
                <Image
                  src={course.thumbnail}
                  alt={course.title}
                  fill
                  className="object-cover"
                  sizes="(max-width: 768px) 100vw, 672px"
                  priority
                />
              ) : (
                <div className="w-full h-full bg-gradient-to-br from-blue-100 to-indigo-200 flex items-center justify-center">
                  <svg
                    className="w-16 h-16 text-blue-300"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth={1}
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25"
                    />
                  </svg>
                </div>
              )}
            </motion.div>
          </div>

          {/* Gradient overlay at bottom of image */}
          <div className="absolute inset-0 bg-gradient-to-t from-black/50 via-transparent to-transparent pointer-events-none" />

          {/* Close button */}
          <button
            onClick={(e) => { e.stopPropagation(); onClose(); }}
            className="absolute top-3 right-3 w-8 h-8 rounded-full bg-black/40 backdrop-blur-sm flex items-center justify-center text-white hover:bg-black/60 transition-colors cursor-pointer z-10"
            aria-label="Close preview"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>

          {/* Badges */}
          {completion > 0 && !isUpcoming && (
            <div className="absolute top-3 left-3 z-10">
              <div className="bg-emerald-500 text-white text-xs font-bold px-2.5 py-1 rounded-full shadow-sm">
                {completion}% complete
              </div>
            </div>
          )}

          {isUpcoming && (
            <div className="absolute top-3 left-3 z-10">
              <div className="bg-amber-500 text-white text-xs font-bold px-2.5 py-1 rounded-full shadow-sm">
                Coming Soon
              </div>
            </div>
          )}
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto flex-1">
          <h2 className="text-xl font-bold text-slate-900 mb-2 leading-snug">
            {course.title}
          </h2>

          <p className="text-base text-slate-600 leading-relaxed mb-4">
            {course.description}
          </p>

          {/* Tags */}
          {course.tags.length > 0 && (
            <div className="flex flex-wrap gap-1.5 mb-4">
              {course.tags.map((tag) => (
                <span
                  key={tag}
                  className="px-2.5 py-1 text-xs font-medium text-slate-500 bg-slate-100 rounded-full"
                >
                  {tag}
                </span>
              ))}
            </div>
          )}

          {/* Meta row */}
          <div className="flex items-center gap-3 mb-5">
            <Badge variant={difficultyVariant[course.difficulty]} size="sm">
              {course.difficulty}
            </Badge>
            <span className="text-sm text-slate-400">~{course.estimatedHours}h</span>
            {course.notebookCount > 0 && (
              <span className="text-sm text-slate-400 flex items-center gap-1">
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M17.25 6.75L22.5 12l-5.25 5.25m-10.5 0L1.5 12l5.25-5.25m7.5-3l-4.5 16.5" />
                </svg>
                {course.notebookCount} notebooks
              </span>
            )}
            {isUpcoming && course.expectedLaunchDate && (
              <span className="text-sm text-amber-600 font-medium">
                {new Date(course.expectedLaunchDate).toLocaleDateString('en-US', {
                  month: 'long',
                  year: 'numeric',
                })}
              </span>
            )}
          </div>

          {/* Progress bar */}
          {completion > 0 && !isUpcoming && (
            <div className="mb-5">
              <div className="w-full h-1.5 bg-slate-100 rounded-full overflow-hidden">
                <div
                  className="h-full bg-emerald-500 rounded-full transition-all duration-500"
                  style={{ width: `${completion}%` }}
                />
              </div>
            </div>
          )}

          {/* CTA */}
          {isUpcoming ? (
            <div className="flex items-center gap-3">
              <NotifyMeButton courseSlug={course.slug} size="md" />
              <span className="text-sm text-slate-400">Get notified when this launches</span>
            </div>
          ) : (
            <button
              onClick={handleCta}
              className="w-full py-3 px-6 bg-accent-blue text-white font-semibold rounded-xl hover:bg-blue-700 active:bg-blue-800 transition-colors cursor-pointer text-base shadow-sm"
            >
              {ctaLabel}
            </button>
          )}
        </div>
      </motion.div>
    </div>
  );
}
