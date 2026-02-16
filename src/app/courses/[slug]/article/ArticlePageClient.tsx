'use client';

import { useEffect } from 'react';
import FadeIn from '@/components/animations/FadeIn';
import Breadcrumb from '@/components/navigation/Breadcrumb';
import CourseNav from '@/components/navigation/CourseNav';
import CourseProgressBar from '@/components/course/CourseProgressBar';
import ArticleReader from '@/components/article/ArticleReader';
import ReadingProgress from '@/components/article/ReadingProgress';
import { markArticleRead } from '@/lib/progress';
import type { NotebookMeta } from '@/types/course';

interface Props {
  slug: string;
  title: string;
  articleContent: string;
  figureUrls: Record<string, string>;
  notebookCount: number;
  notebooks: NotebookMeta[];
  hasCaseStudy: boolean;
}

export default function ArticlePageClient({
  slug,
  title,
  articleContent,
  figureUrls,
  notebookCount,
  notebooks,
  hasCaseStudy,
}: Props) {
  // Mark article as read after spending some time on the page
  useEffect(() => {
    const timer = setTimeout(() => {
      markArticleRead(slug);
    }, 10000); // 10 seconds
    return () => clearTimeout(timer);
  }, [slug]);

  return (
    <>
      <CourseProgressBar
        courseSlug={slug}
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
              { label: title, href: `/courses/${slug}` },
              { label: 'Article' },
            ]}
          />
        </FadeIn>

        <FadeIn delay={0.1}>
          <ArticleReader content={articleContent} figureUrls={figureUrls} courseSlug={slug} />
        </FadeIn>

        <FadeIn delay={0.15}>
          <CourseNav
            prevHref={`/courses/${slug}`}
            prevLabel="Course Overview"
            nextHref={notebookCount > 0 ? `/courses/${slug}/practice` : undefined}
            nextLabel={notebookCount > 0 ? "Continue to Practice" : undefined}
          />
        </FadeIn>
      </div>
    </>
  );
}
