'use client';

import Link from 'next/link';
import FadeIn from '@/components/animations/FadeIn';
import Breadcrumb from '@/components/navigation/Breadcrumb';
import Badge from '@/components/ui/Badge';
import Button from '@/components/ui/Button';
import NotifyMeButton from '@/components/catalog/NotifyMeButton';
import type { CourseCard } from '@/types/course';

interface Props {
  course: CourseCard;
}

export default function UpcomingCourseClient({ course }: Props) {
  const launchDate = course.expectedLaunchDate
    ? new Date(course.expectedLaunchDate).toLocaleDateString('en-US', {
        month: 'long',
        day: 'numeric',
        year: 'numeric',
      })
    : 'TBD';

  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 py-8">
      <FadeIn>
        <Breadcrumb
          items={[
            { label: 'Courses', href: '/' },
            { label: course.title },
          ]}
        />
      </FadeIn>

      <FadeIn delay={0.1}>
        <div className="mb-6">
          <div className="flex items-center gap-3 mb-4">
            <Badge variant="amber" size="md">Coming Soon</Badge>
            <span className="text-sm text-text-muted">Expected {launchDate}</span>
          </div>

          <h1 className="text-3xl sm:text-4xl font-bold text-foreground tracking-tight mb-4">
            {course.title}
          </h1>

          <p className="text-text-secondary leading-relaxed mb-6 text-lg">
            {course.description}
          </p>

          <div className="flex items-center gap-3 flex-wrap mb-8">
            <Badge
              variant={course.difficulty === 'beginner' ? 'green' : course.difficulty === 'advanced' ? 'amber' : 'blue'}
              size="md"
            >
              {course.difficulty}
            </Badge>
            <span className="text-sm text-text-muted">~{course.estimatedHours} hours</span>
            {course.tags.map((tag) => (
              <span key={tag} className="text-xs px-2.5 py-1 rounded-full bg-gray-100 text-text-secondary">
                {tag}
              </span>
            ))}
          </div>
        </div>
      </FadeIn>

      <FadeIn delay={0.2}>
        <div className="bg-gradient-to-br from-amber-50 to-orange-50/30 border border-amber-200/60 rounded-2xl p-8 mb-8">
          <div className="text-center">
            <div className="w-16 h-16 bg-gradient-to-br from-amber-400 to-orange-500 rounded-2xl flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M14.857 17.082a23.848 23.848 0 005.454-1.31A8.967 8.967 0 0118 9.75v-.7V9A6 6 0 006 9v.75a8.967 8.967 0 01-2.312 6.022c1.733.64 3.56 1.085 5.455 1.31m5.714 0a24.255 24.255 0 01-5.714 0m5.714 0a3 3 0 11-5.714 0" />
              </svg>
            </div>
            <h2 className="text-xl font-bold text-foreground mb-2">
              Get notified when this course launches
            </h2>
            <p className="text-sm text-text-secondary mb-6 max-w-md mx-auto">
              We&apos;ll send you a notification as soon as this course is available. Be the first to start learning!
            </p>
            <NotifyMeButton courseSlug={course.slug} size="md" />
          </div>
        </div>
      </FadeIn>

      <FadeIn delay={0.3}>
        <Link href="/">
          <Button variant="ghost">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 19.5L3 12m0 0l7.5-7.5M3 12h18" />
            </svg>
            Back to Home
          </Button>
        </Link>
      </FadeIn>
    </div>
  );
}
