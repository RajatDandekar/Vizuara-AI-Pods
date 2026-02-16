'use client';

import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { motion, AnimatePresence } from 'framer-motion';
import Button from '@/components/ui/Button';
import Badge from '@/components/ui/Badge';
import type { CourseCard } from '@/types/course';

const difficultyVariant: Record<string, 'green' | 'blue' | 'amber'> = {
  beginner: 'green',
  intermediate: 'blue',
  advanced: 'amber',
};

interface HeroCarouselProps {
  courses: CourseCard[];
}

export default function HeroCarousel({ courses }: HeroCarouselProps) {
  const [current, setCurrent] = useState(0);

  const next = useCallback(() => {
    setCurrent((prev) => (prev + 1) % courses.length);
  }, [courses.length]);

  useEffect(() => {
    const timer = setInterval(next, 6000);
    return () => clearInterval(timer);
  }, [next]);

  if (courses.length === 0) return null;

  const course = courses[current];

  return (
    <div className="relative rounded-2xl overflow-hidden bg-gradient-to-br from-slate-900 via-blue-950 to-indigo-900 min-h-[320px] sm:min-h-[360px]">
      {/* Background image */}
      <AnimatePresence mode="wait">
        <motion.div
          key={course.slug}
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.15 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.6 }}
          className="absolute inset-0"
        >
          {course.thumbnail && (
            <Image
              src={course.thumbnail}
              alt=""
              fill
              className="object-cover"
              sizes="100vw"
            />
          )}
        </motion.div>
      </AnimatePresence>

      {/* Gradient overlay */}
      <div className="absolute inset-0 bg-gradient-to-r from-slate-900/95 via-slate-900/80 to-slate-900/40" />

      {/* Content */}
      <div className="relative z-10 flex flex-col justify-center h-full min-h-[320px] sm:min-h-[360px] px-8 sm:px-12 py-10">
        <AnimatePresence mode="wait">
          <motion.div
            key={course.slug}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.4, ease: [0.25, 0.1, 0.25, 1] }}
          >
            <div className="flex items-center gap-2 mb-4">
              <Badge variant={difficultyVariant[course.difficulty]} size="sm">
                {course.difficulty}
              </Badge>
              <span className="text-white/50 text-xs">~{course.estimatedHours}h</span>
              {course.notebookCount > 0 && (
                <span className="text-white/50 text-xs flex items-center gap-1">
                  <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M17.25 6.75L22.5 12l-5.25 5.25m-10.5 0L1.5 12l5.25-5.25m7.5-3l-4.5 16.5" />
                  </svg>
                  {course.notebookCount} notebooks
                </span>
              )}
            </div>

            <h2 className="text-2xl sm:text-3xl lg:text-4xl font-bold text-white mb-3 max-w-xl leading-tight">
              {course.title}
            </h2>

            <p className="text-white/70 text-sm sm:text-base leading-relaxed mb-6 max-w-lg line-clamp-2">
              {course.description}
            </p>

            <Link href={`/courses/${course.slug}`}>
              <Button size="lg">
                Start Learning
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
                </svg>
              </Button>
            </Link>
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Carousel dots */}
      {courses.length > 1 && (
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-2 z-10">
          {courses.map((_, i) => (
            <button
              key={i}
              onClick={() => setCurrent(i)}
              className={`rounded-full transition-all duration-300 cursor-pointer ${
                i === current
                  ? 'w-6 h-2 bg-white'
                  : 'w-2 h-2 bg-white/40 hover:bg-white/60'
              }`}
            />
          ))}
        </div>
      )}
    </div>
  );
}
