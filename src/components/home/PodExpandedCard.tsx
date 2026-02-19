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
import type { PodCard } from '@/types/course';

interface PodExpandedCardProps {
  pod: PodCard;
  courseSlug: string;
  courseTitle: string;
  onClose: () => void;
  onMouseEnter: () => void;
  onMouseLeave: () => void;
}

const ease = [0.25, 0.1, 0.25, 1] as [number, number, number, number];

export default function PodExpandedCard({
  pod,
  courseSlug,
  courseTitle,
  onClose,
  onMouseEnter,
  onMouseLeave,
}: PodExpandedCardProps) {
  const router = useRouter();

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

  function handleCta(e: React.MouseEvent) {
    e.stopPropagation();
    onClose();
    router.push(`/courses/${courseSlug}/${pod.slug}`);
  }

  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center p-4 pointer-events-none">
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
          <div className="absolute inset-0 kenburns-active">
            <motion.div
              className="relative w-full h-full"
              style={{ x: imgX, y: imgY, scale: 1.15 }}
            >
              {pod.thumbnail ? (
                <Image
                  src={pod.thumbnail}
                  alt={pod.title}
                  fill
                  className="object-cover"
                  sizes="(max-width: 768px) 100vw, 672px"
                  priority
                />
              ) : (
                <div className="w-full h-full bg-gradient-to-br from-blue-100 to-indigo-200 flex items-center justify-center">
                  <span className="text-5xl font-bold text-blue-300/60">{pod.order}</span>
                </div>
              )}
            </motion.div>
          </div>

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

          {/* Pod order badge */}
          <div className="absolute top-3 left-3 z-10">
            <div className="bg-blue-600 text-white text-xs font-bold px-2.5 py-1 rounded-full shadow-sm">
              Pod {pod.order}
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto flex-1">
          {/* Course breadcrumb */}
          <p className="text-xs font-medium text-blue-600 mb-1.5">{courseTitle}</p>

          <h2 className="text-xl font-bold text-slate-900 mb-2 leading-snug">
            {pod.title}
          </h2>

          <p className="text-base text-slate-600 leading-relaxed mb-4">
            {pod.description}
          </p>

          {/* Meta row */}
          <div className="flex items-center gap-3 flex-wrap mb-5">
            <span className="text-sm text-slate-400">~{pod.estimatedHours}h</span>
            {pod.notebookCount > 0 && (
              <span className="text-sm text-slate-400 flex items-center gap-1">
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M17.25 6.75L22.5 12l-5.25 5.25m-10.5 0L1.5 12l5.25-5.25m7.5-3l-4.5 16.5" />
                </svg>
                {pod.notebookCount} notebook{pod.notebookCount !== 1 ? 's' : ''}
              </span>
            )}
            {pod.hasCaseStudy && (
              <span className="text-sm text-slate-400 flex items-center gap-1">
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
                </svg>
                Case study included
              </span>
            )}
          </div>

          {/* What you'll learn */}
          <div className="bg-slate-50 rounded-xl p-4 mb-5">
            <h3 className="text-sm font-semibold text-slate-700 mb-2">What you&apos;ll get</h3>
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-sm text-slate-600">
                <div className="w-5 h-5 rounded-full bg-blue-100 flex items-center justify-center flex-shrink-0">
                  <svg className="w-3 h-3 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
                  </svg>
                </div>
                In-depth article with figures and equations
              </div>
              {pod.notebookCount > 0 && (
                <div className="flex items-center gap-2 text-sm text-slate-600">
                  <div className="w-5 h-5 rounded-full bg-emerald-100 flex items-center justify-center flex-shrink-0">
                    <svg className="w-3 h-3 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
                    </svg>
                  </div>
                  {pod.notebookCount} hands-on Colab notebook{pod.notebookCount !== 1 ? 's' : ''} with TODO exercises
                </div>
              )}
              {pod.hasCaseStudy && (
                <div className="flex items-center gap-2 text-sm text-slate-600">
                  <div className="w-5 h-5 rounded-full bg-amber-100 flex items-center justify-center flex-shrink-0">
                    <svg className="w-3 h-3 text-amber-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
                    </svg>
                  </div>
                  Industry case study with production constraints
                </div>
              )}
            </div>
          </div>

          {/* CTA */}
          <button
            onClick={handleCta}
            className="w-full py-3 px-6 bg-accent-blue text-white font-semibold rounded-xl hover:bg-blue-700 active:bg-blue-800 transition-colors cursor-pointer text-base shadow-sm"
          >
            Start Learning
          </button>
        </div>
      </motion.div>
    </div>
  );
}
