import { getPod, getAllPodParams, getCourseManifest, getPodCaseStudyContent, parseCaseStudySections } from '@/lib/content';
import { notFound } from 'next/navigation';
import AuthGate from '@/components/auth/AuthGate';
import CaseStudyPageClient from './CaseStudyPageClient';

export function generateStaticParams() {
  return getAllPodParams().map(({ courseSlug, podSlug }) => ({ courseSlug, podSlug }));
}

interface PageProps {
  params: Promise<{ courseSlug: string; podSlug: string }>;
}

export default async function CaseStudyPage({ params }: PageProps) {
  const { courseSlug, podSlug } = await params;

  try {
    const pod = getPod(courseSlug, podSlug);
    const course = getCourseManifest(courseSlug);

    if (!pod.caseStudy) {
      notFound();
    }

    const content = getPodCaseStudyContent(courseSlug, podSlug);
    if (!content) {
      notFound();
    }

    const sections = parseCaseStudySections(content);

    return (
      <AuthGate>
        <CaseStudyPageClient
          courseSlug={courseSlug}
          courseTitle={course.title}
          podSlug={podSlug}
          podTitle={pod.title}
          caseStudy={pod.caseStudy}
          sections={sections}
          notebooks={pod.notebooks}
        />
      </AuthGate>
    );
  } catch {
    notFound();
  }
}
