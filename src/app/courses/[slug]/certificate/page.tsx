import { getCourse, getCourseSlugs } from '@/lib/content';
import { notFound } from 'next/navigation';
import AuthGate from '@/components/auth/AuthGate';
import CertificatePageClient from './CertificatePageClient';

export function generateStaticParams() {
  return getCourseSlugs().map((slug) => ({ slug }));
}

interface PageProps {
  params: Promise<{ slug: string }>;
}

export default async function CertificatePage({ params }: PageProps) {
  const { slug } = await params;

  try {
    const course = getCourse(slug);

    return (
      <AuthGate>
        <CertificatePageClient
          slug={course.slug}
          courseTitle={course.title}
          difficulty={course.difficulty}
          estimatedHours={course.estimatedHours}
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
