import { getCourse, getCourseSlugs, getCaseStudyContent, parseCaseStudySections } from '@/lib/content';
import { notFound } from 'next/navigation';
import AuthGate from '@/components/auth/AuthGate';
import CaseStudyPageClient from './CaseStudyPageClient';

export function generateStaticParams() {
  return getCourseSlugs().map((slug) => ({ slug }));
}

interface PageProps {
  params: Promise<{ slug: string }>;
}

export default async function CaseStudyPage({ params }: PageProps) {
  const { slug } = await params;

  try {
    const course = getCourse(slug);

    if (!course.caseStudy) {
      notFound();
    }

    const content = getCaseStudyContent(slug);
    if (!content) {
      notFound();
    }

    const sections = parseCaseStudySections(content);

    return (
      <AuthGate>
        <CaseStudyPageClient
          slug={slug}
          courseTitle={course.title}
          caseStudy={course.caseStudy}
          sections={sections}
          notebooks={course.notebooks}
        />
      </AuthGate>
    );
  } catch {
    notFound();
  }
}
