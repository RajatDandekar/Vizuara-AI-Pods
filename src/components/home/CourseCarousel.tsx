'use client';

import { useRef, useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { AnimatePresence } from 'framer-motion';
import CourseExpandedCard from './CourseExpandedCard';
import type { CourseCard } from '@/types/course';

interface CourseCarouselProps {
  courses: CourseCard[];
  completions?: Record<string, number>;
}

const difficultyColor: Record<string, string> = {
  beginner: 'bg-emerald-50 text-emerald-700 border-emerald-200',
  intermediate: 'bg-blue-50 text-blue-700 border-blue-200',
  advanced: 'bg-amber-50 text-amber-700 border-amber-200',
};

const HOVER_OPEN_DELAY = 400;   // ms before expanded card appears
const HOVER_CLOSE_DELAY = 300;  // ms grace period when moving between card and overlay

export default function CourseCarousel({ courses, completions = {} }: CourseCarouselProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [canScrollLeft, setCanScrollLeft] = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(false);

  // --- Hover expansion state ---
  const [expandedSlug, setExpandedSlug] = useState<string | null>(null);
  const openTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const closeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Clean up timers on unmount
  useEffect(() => {
    return () => {
      if (openTimerRef.current) clearTimeout(openTimerRef.current);
      if (closeTimerRef.current) clearTimeout(closeTimerRef.current);
    };
  }, []);

  function handleCardMouseEnter(slug: string) {
    // Cancel any pending close
    if (closeTimerRef.current) {
      clearTimeout(closeTimerRef.current);
      closeTimerRef.current = null;
    }
    // If already showing this card, do nothing
    if (expandedSlug === slug) return;
    // Cancel any pending open for a different card
    if (openTimerRef.current) {
      clearTimeout(openTimerRef.current);
    }
    // Start open timer
    openTimerRef.current = setTimeout(() => {
      setExpandedSlug(slug);
      openTimerRef.current = null;
    }, HOVER_OPEN_DELAY);
  }

  function handleCardMouseLeave() {
    // Cancel pending open
    if (openTimerRef.current) {
      clearTimeout(openTimerRef.current);
      openTimerRef.current = null;
    }
    // Start close timer (grace period to move to overlay)
    if (expandedSlug) {
      closeTimerRef.current = setTimeout(() => {
        setExpandedSlug(null);
        closeTimerRef.current = null;
      }, HOVER_CLOSE_DELAY);
    }
  }

  function handleOverlayMouseEnter() {
    // Mouse entered the expanded overlay — cancel close
    if (closeTimerRef.current) {
      clearTimeout(closeTimerRef.current);
      closeTimerRef.current = null;
    }
  }

  function handleOverlayMouseLeave() {
    // Mouse left the expanded overlay — close it
    closeTimerRef.current = setTimeout(() => {
      setExpandedSlug(null);
      closeTimerRef.current = null;
    }, HOVER_CLOSE_DELAY);
  }

  function handleOverlayClose() {
    if (openTimerRef.current) clearTimeout(openTimerRef.current);
    if (closeTimerRef.current) clearTimeout(closeTimerRef.current);
    setExpandedSlug(null);
  }

  // --- Scroll logic (unchanged) ---
  const checkScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    setCanScrollLeft(el.scrollLeft > 4);
    setCanScrollRight(el.scrollLeft < el.scrollWidth - el.clientWidth - 4);
  }, []);

  useEffect(() => {
    checkScroll();
    const el = scrollRef.current;
    if (!el) return;
    el.addEventListener('scroll', checkScroll, { passive: true });
    const ro = new ResizeObserver(checkScroll);
    ro.observe(el);
    return () => {
      el.removeEventListener('scroll', checkScroll);
      ro.disconnect();
    };
  }, [checkScroll, courses]);

  const scroll = (direction: 'left' | 'right') => {
    const el = scrollRef.current;
    if (!el) return;
    const amount = el.clientWidth * 0.85;
    el.scrollBy({ left: direction === 'right' ? amount : -amount, behavior: 'smooth' });
  };

  if (courses.length === 0) return null;

  return (
    <div className="relative group/carousel">
      {/* Left arrow */}
      {canScrollLeft && (
        <button
          onClick={() => scroll('left')}
          className="absolute left-0 top-1/2 -translate-y-1/2 z-10 w-10 h-10 rounded-full bg-white border border-slate-200 shadow-lg flex items-center justify-center text-slate-600 hover:text-slate-900 hover:border-slate-300 transition-all opacity-0 group-hover/carousel:opacity-100 -translate-x-1/2 cursor-pointer"
          aria-label="Scroll left"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 19.5L8.25 12l7.5-7.5" />
          </svg>
        </button>
      )}

      {/* Right arrow */}
      {canScrollRight && (
        <button
          onClick={() => scroll('right')}
          className="absolute right-0 top-1/2 -translate-y-1/2 z-10 w-10 h-10 rounded-full bg-white border border-slate-200 shadow-lg flex items-center justify-center text-slate-600 hover:text-slate-900 hover:border-slate-300 transition-all opacity-0 group-hover/carousel:opacity-100 translate-x-1/2 cursor-pointer"
          aria-label="Scroll right"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
          </svg>
        </button>
      )}

      {/* Scrollable row */}
      <div
        ref={scrollRef}
        className="flex gap-4 overflow-x-auto scrollbar-hide scroll-smooth pb-1"
      >
        {courses.map((course) => {
          const completion = completions[course.slug] || 0;
          return (
            <div
              key={course.slug}
              className="flex-shrink-0 w-[280px] sm:w-[320px]"
              onMouseEnter={() => handleCardMouseEnter(course.slug)}
              onMouseLeave={handleCardMouseLeave}
            >
              <Link
                href={course.status === 'upcoming' ? '#' : `/courses/${course.slug}`}
              >
                <div className="bg-white rounded-2xl overflow-hidden h-full flex flex-col cursor-pointer group border border-blue-200 shadow-sm hover:shadow-lg hover:border-blue-300 hover:-translate-y-1 transition-all duration-300">
                  {/* Thumbnail */}
                  {course.thumbnail ? (
                    <div className="relative w-full aspect-[16/9] overflow-hidden">
                      <Image
                        src={course.thumbnail}
                        alt={course.title}
                        fill
                        className="object-cover transition-transform duration-500 group-hover:scale-110"
                        sizes="280px"
                      />
                      <div className="absolute inset-0 bg-gradient-to-t from-white via-white/20 to-transparent" />

                      {completion > 0 && (
                        <div className="absolute top-2.5 right-2.5">
                          <div className="bg-emerald-50 backdrop-blur-sm border border-emerald-200 rounded-full px-2 py-0.5 text-[11px] font-bold text-emerald-700">
                            {completion}%
                          </div>
                        </div>
                      )}

                      {course.status === 'upcoming' && (
                        <div className="absolute top-2.5 left-2.5">
                          <div className="bg-amber-50 backdrop-blur-sm border border-amber-200 rounded-full px-2 py-0.5 text-[11px] font-medium text-amber-700">
                            Coming Soon
                          </div>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="relative w-full aspect-[16/9] bg-gradient-to-br from-blue-50 to-indigo-50 flex items-center justify-center">
                      <svg className="w-8 h-8 text-blue-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" />
                      </svg>
                      {course.status === 'upcoming' && (
                        <div className="absolute top-2.5 left-2.5">
                          <div className="bg-amber-50 backdrop-blur-sm border border-amber-200 rounded-full px-2 py-0.5 text-[11px] font-medium text-amber-700">
                            Coming Soon
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Content */}
                  <div className="p-4 flex flex-col flex-1">
                    <h3 className="font-bold text-slate-900 text-lg leading-snug mb-1.5 group-hover:text-blue-600 transition-colors line-clamp-2">
                      {course.title}
                    </h3>

                    <p className="text-base text-slate-700 leading-relaxed mb-3 line-clamp-2">
                      {course.description}
                    </p>

                    {/* Meta — pushed to bottom */}
                    <div className="flex items-center gap-2 mt-auto pt-2 border-t border-slate-100">
                      <span className={`text-[11px] font-medium px-2 py-0.5 rounded-full border ${difficultyColor[course.difficulty]}`}>
                        {course.difficulty}
                      </span>
                      <span className="text-[11px] text-slate-400">~{course.estimatedHours}h</span>
                      {course.podCount > 0 && (
                        <span className="text-[11px] text-slate-400 ml-auto">
                          {course.podCount} pod{course.podCount !== 1 ? 's' : ''}
                        </span>
                      )}
                      {(course.totalNotebooks || course.notebookCount || 0) > 0 && (
                        <span className="text-[11px] text-slate-400 flex items-center gap-1">
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
            </div>
          );
        })}
      </div>

      {/* Expanded card overlay — triggered by hover */}
      <AnimatePresence>
        {expandedSlug && (() => {
          const course = courses.find((c) => c.slug === expandedSlug);
          if (!course) return null;
          return (
            <CourseExpandedCard
              key={course.slug}
              course={course}
              completion={completions[course.slug] || 0}
              onClose={handleOverlayClose}
              onMouseEnter={handleOverlayMouseEnter}
              onMouseLeave={handleOverlayMouseLeave}
            />
          );
        })()}
      </AnimatePresence>
    </div>
  );
}
