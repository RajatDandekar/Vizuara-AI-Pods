'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import FadeIn from '@/components/animations/FadeIn';
import Breadcrumb from '@/components/navigation/Breadcrumb';
import Badge from '@/components/ui/Badge';
import Button from '@/components/ui/Button';
import type { PodCard } from '@/types/course';
import { getCourseCompletion, getPodCompletion, isCourseComplete, onProgressChange } from '@/lib/progress';

const difficultyVariant: Record<string, 'green' | 'blue' | 'amber'> = {
  beginner: 'green',
  intermediate: 'blue',
  advanced: 'amber',
};

interface Props {
  courseSlug: string;
  title: string;
  description: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  estimatedHours: number;
  tags: string[];
  allPods: PodCard[];
  livePods: PodCard[];
  status?: string;
}

export default function CourseOverviewClient({
  courseSlug,
  title,
  description,
  difficulty,
  estimatedHours,
  tags,
  allPods,
  livePods,
  status,
}: Props) {
  const [courseProgress, setCourseProgress] = useState({ completedPods: 0, totalPods: 0, percentage: 0 });
  const [podCompletions, setPodCompletions] = useState<Record<string, number>>({});

  useEffect(() => {
    function refresh() {
      setCourseProgress(getCourseCompletion(courseSlug, livePods));
      const comps: Record<string, number> = {};
      for (const pod of livePods) {
        comps[pod.slug] = getPodCompletion(courseSlug, pod.slug, pod.notebookCount, pod.hasCaseStudy);
      }
      setPodCompletions(comps);
    }
    refresh();
    return onProgressChange(refresh);
  }, [courseSlug, livePods]);

  const courseComplete = isCourseComplete(courseSlug, livePods);

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 py-8">
      <FadeIn>
        <Breadcrumb
          items={[
            { label: 'Courses', href: '/' },
            { label: title },
          ]}
        />
      </FadeIn>

      {/* Course Header */}
      <FadeIn delay={0.1}>
        <div className="mb-8">
          <h1 className="text-3xl sm:text-4xl font-bold text-foreground tracking-tight mb-4">
            {title}
          </h1>
          <p className="text-text-secondary leading-relaxed mb-5 max-w-3xl">
            {description}
          </p>
          <div className="flex items-center gap-3 flex-wrap">
            <Badge variant={difficultyVariant[difficulty]} size="md">
              {difficulty}
            </Badge>
            <span className="text-sm text-text-muted flex items-center gap-1.5">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              ~{estimatedHours} hours
            </span>
            <span className="text-sm text-text-muted">
              {livePods.length} pod{livePods.length !== 1 ? 's' : ''} live
              {allPods.length > livePods.length && (
                <span className="text-text-muted/60"> / {allPods.length} total</span>
              )}
            </span>
          </div>
        </div>
      </FadeIn>

      {/* Course Progress */}
      {courseProgress.percentage > 0 && (
        <FadeIn delay={0.15}>
          <div className="bg-card-bg border border-card-border rounded-xl p-4 mb-8">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-foreground">Course Progress</span>
              <span className="text-sm text-text-muted">
                {courseProgress.completedPods}/{courseProgress.totalPods} pods complete
              </span>
            </div>
            <div className="w-full h-2 bg-gray-100 rounded-full overflow-hidden">
              <div
                className="h-full bg-accent-green rounded-full transition-all duration-500"
                style={{ width: `${courseProgress.percentage}%` }}
              />
            </div>
          </div>
        </FadeIn>
      )}

      {/* Course Certificate CTA */}
      {courseComplete && livePods.length > 1 && (
        <FadeIn delay={0.18}>
          <Link href={`/courses/${courseSlug}/certificate`}>
            <div className="bg-gradient-to-br from-amber-50 to-yellow-50/40 border border-amber-200/60 rounded-xl p-5 mb-8 hover:shadow-md hover:border-amber-300/60 transition-all cursor-pointer">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-amber-400 to-yellow-500 flex items-center justify-center flex-shrink-0">
                  <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 18.75h-9m9 0a3 3 0 013 3h-15a3 3 0 013-3m9 0v-3.375c0-.621-.503-1.125-1.125-1.125h-.871M7.5 18.75v-3.375c0-.621.504-1.125 1.125-1.125h.872m5.007 0H9.497m5.007 0a7.454 7.454 0 01-.982-3.172M9.497 14.25a7.454 7.454 0 00.981-3.172M5.25 4.236c-.982.143-1.954.317-2.916.52A6.003 6.003 0 007.73 9.728M5.25 4.236V4.5c0 2.108.966 3.99 2.48 5.228M5.25 4.236V2.721C7.456 2.41 9.71 2.25 12 2.25c2.291 0 4.545.16 6.75.47v1.516M18.75 4.236c.982.143 1.954.317 2.916.52A6.003 6.003 0 0016.27 9.728M18.75 4.236V4.5c0 2.108-.966 3.99-2.48 5.228m0 0a6.003 6.003 0 01-5.54 0" />
                  </svg>
                </div>
                <div className="flex-1 min-w-0">
                  <h4 className="font-semibold text-foreground text-sm">Course Certificate Available!</h4>
                  <p className="text-xs text-text-secondary mt-0.5">
                    You&apos;ve completed all pods. Claim your course certificate.
                  </p>
                </div>
                <svg className="w-4 h-4 text-text-muted flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
                </svg>
              </div>
            </div>
          </Link>
        </FadeIn>
      )}

      {/* Pod Grid */}
      <FadeIn delay={0.2}>
        <h2 className="text-sm font-semibold text-text-muted uppercase tracking-wider mb-4">
          Pods in this Course
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {allPods.map((pod) => {
            const isLive = livePods.some((lp) => lp.slug === pod.slug);
            const completion = podCompletions[pod.slug] || 0;

            if (!isLive) {
              return (
                <div
                  key={pod.slug}
                  className="bg-card-bg border border-card-border rounded-xl p-5 opacity-60"
                >
                  {/* Thumbnail */}
                  {pod.thumbnail && (
                    <div className="relative w-full aspect-[16/9] rounded-lg overflow-hidden mb-3 bg-gray-50">
                      <Image
                        src={pod.thumbnail}
                        alt={pod.title}
                        fill
                        className="object-cover"
                        sizes="(max-width: 640px) 100vw, 320px"
                      />
                    </div>
                  )}
                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 rounded-lg bg-gray-100 text-text-muted flex items-center justify-center flex-shrink-0 text-sm font-semibold">
                      {pod.order}
                    </div>
                    <div className="flex-1 min-w-0">
                      <h3 className="font-semibold text-foreground text-sm mb-1">{pod.title}</h3>
                      <p className="text-xs text-text-muted">Coming soon</p>
                    </div>
                  </div>
                </div>
              );
            }

            return (
              <Link key={pod.slug} href={`/courses/${courseSlug}/${pod.slug}`}>
                <div className="bg-card-bg border border-card-border rounded-xl p-5 hover:shadow-md hover:border-accent-blue/30 transition-all cursor-pointer h-full">
                  {/* Thumbnail */}
                  {pod.thumbnail && (
                    <div className="relative w-full aspect-[16/9] rounded-lg overflow-hidden mb-3 bg-gray-50">
                      <Image
                        src={pod.thumbnail}
                        alt={pod.title}
                        fill
                        className="object-cover"
                        sizes="(max-width: 640px) 100vw, 320px"
                      />
                    </div>
                  )}

                  <div className="flex items-start gap-3">
                    <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 text-sm font-semibold ${
                      completion >= 100
                        ? 'bg-accent-green-light text-accent-green'
                        : 'bg-blue-50 text-accent-blue'
                    }`}>
                      {completion >= 100 ? (
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
                        </svg>
                      ) : (
                        pod.order
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <h3 className="font-semibold text-foreground text-sm mb-1">{pod.title}</h3>
                      <p className="text-xs text-text-secondary line-clamp-2">{pod.description}</p>
                      <div className="flex items-center gap-3 mt-2">
                        <span className="text-xs text-text-muted">~{pod.estimatedHours}h</span>
                        {pod.notebookCount > 0 && (
                          <span className="text-xs text-text-muted">{pod.notebookCount} notebooks</span>
                        )}
                        {pod.hasCaseStudy && (
                          <span className="text-xs text-text-muted">Case study</span>
                        )}
                      </div>
                      {/* Pod progress bar */}
                      {completion > 0 && (
                        <div className="mt-2">
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
                </div>
              </Link>
            );
          })}
        </div>
      </FadeIn>
    </div>
  );
}
