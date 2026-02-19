'use client';

import { useEffect } from 'react';
import FadeIn from '@/components/animations/FadeIn';
import Breadcrumb from '@/components/navigation/Breadcrumb';
import CourseNav from '@/components/navigation/CourseNav';
import CourseProgressBar from '@/components/course/CourseProgressBar';
import ArticleReader from '@/components/article/ArticleReader';
import ReadingProgress from '@/components/article/ReadingProgress';
import { markPodArticleRead } from '@/lib/progress';
import type { NotebookMeta } from '@/types/course';

interface Props {
  courseSlug: string;
  courseTitle: string;
  podSlug: string;
  title: string;
  articleContent: string;
  figureUrls: Record<string, string>;
  notebookCount: number;
  notebooks: NotebookMeta[];
  hasCaseStudy: boolean;
}

export default function ArticlePageClient({
  courseSlug,
  courseTitle,
  podSlug,
  title,
  articleContent,
  figureUrls,
  notebookCount,
  notebooks,
  hasCaseStudy,
}: Props) {
  useEffect(() => {
    const timer = setTimeout(() => {
      markPodArticleRead(courseSlug, podSlug);
    }, 10000);
    return () => clearTimeout(timer);
  }, [courseSlug, podSlug]);

  const basePath = `/courses/${courseSlug}/${podSlug}`;

  return (
    <>
      <CourseProgressBar
        courseSlug={courseSlug}
        podSlug={podSlug}
        courseTitle={title}
        notebooks={notebooks}
        hasCaseStudy={hasCaseStudy}
        activeStep="article"
      />
      <ReadingProgress />
      <div className="max-w-3xl mx-auto px-4 sm:px-6 py-8">
        <FadeIn>
          <Breadcrumb
            items={[
              { label: 'Courses', href: '/' },
              { label: courseTitle, href: `/courses/${courseSlug}` },
              { label: title, href: basePath },
              { label: 'Article' },
            ]}
          />
        </FadeIn>

        <FadeIn delay={0.1}>
          <ArticleReader content={articleContent} figureUrls={figureUrls} courseSlug={`${courseSlug}/pods/${podSlug}`} />
        </FadeIn>

        <FadeIn delay={0.15}>
          <CourseNav
            prevHref={basePath}
            prevLabel="Pod Overview"
            nextHref={notebookCount > 0 ? `${basePath}/practice` : undefined}
            nextLabel={notebookCount > 0 ? "Continue to Practice" : undefined}
          />
        </FadeIn>
      </div>
    </>
  );
}
