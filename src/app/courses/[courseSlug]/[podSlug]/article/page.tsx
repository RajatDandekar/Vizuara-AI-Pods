import { getPod, getAllPodParams, getCourseManifest } from '@/lib/content';
import { notFound } from 'next/navigation';
import SubscriptionGate from '@/components/auth/SubscriptionGate';
import ArticlePageClient from './ArticlePageClient';

export function generateStaticParams() {
  return getAllPodParams().map(({ courseSlug, podSlug }) => ({ courseSlug, podSlug }));
}

interface PageProps {
  params: Promise<{ courseSlug: string; podSlug: string }>;
}

export default async function ArticlePage({ params }: PageProps) {
  const { courseSlug, podSlug } = await params;

  try {
    const pod = getPod(courseSlug, podSlug);
    const course = getCourseManifest(courseSlug);
    return (
      <SubscriptionGate podSlug={podSlug}>
        <ArticlePageClient
          courseSlug={courseSlug}
          courseTitle={course.title}
          podSlug={pod.slug}
          title={pod.title}
          articleContent={pod.articleContent}
          figureUrls={pod.article.figureUrls}
          notebookCount={pod.notebooks.length}
          notebooks={pod.notebooks}
          hasCaseStudy={!!pod.caseStudy}
        />
      </SubscriptionGate>
    );
  } catch {
    notFound();
  }
}
