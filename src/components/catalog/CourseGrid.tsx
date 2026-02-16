'use client';

import StaggerChildren from '@/components/animations/StaggerChildren';
import CourseCard from './CourseCard';
import type { CourseCard as CourseCardType } from '@/types/course';

interface CourseGridProps {
  courses: CourseCardType[];
  completions?: Record<string, number>;
}

export default function CourseGrid({ courses, completions = {} }: CourseGridProps) {
  if (courses.length === 0) {
    return (
      <div className="text-center py-16">
        <p className="text-text-muted text-sm">No courses match your filter.</p>
      </div>
    );
  }

  return (
    <StaggerChildren
      staggerDelay={0.1}
      className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6"
    >
      {courses.map((course) => (
        <CourseCard
          key={course.slug}
          course={course}
          completion={completions[course.slug] || 0}
        />
      ))}
    </StaggerChildren>
  );
}
