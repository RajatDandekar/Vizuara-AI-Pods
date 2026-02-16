import { getCourse, getCourseSlugs } from '@/lib/content';
import { notFound } from 'next/navigation';
import AuthGate from '@/components/auth/AuthGate';
import ArticlePageClient from './ArticlePageClient';

export function generateStaticParams() {
  return getCourseSlugs().map((slug) => ({ slug }));
}

interface PageProps {
  params: Promise<{ slug: string }>;
}

export default async function ArticlePage({ params }: PageProps) {
  const { slug } = await params;

  try {
    const course = getCourse(slug);
    return (
      <AuthGate>
        <ArticlePageClient
          slug={course.slug}
          title={course.title}
          articleContent={course.articleContent}
          figureUrls={course.article.figureUrls}
          notebookCount={course.notebooks.length}
          notebooks={course.notebooks}
          hasCaseStudy={!!course.caseStudy}
        />
      </AuthGate>
    );
  } catch {
    notFound();
  }
}
