import { getPod, getAllPodParams, getCourseManifest } from '@/lib/content';
import { notFound } from 'next/navigation';
import AuthGate from '@/components/auth/AuthGate';
import CertificatePageClient from './CertificatePageClient';

export function generateStaticParams() {
  return getAllPodParams().map(({ courseSlug, podSlug }) => ({ courseSlug, podSlug }));
}

interface PageProps {
  params: Promise<{ courseSlug: string; podSlug: string }>;
}

export default async function CertificatePage({ params }: PageProps) {
  const { courseSlug, podSlug } = await params;

  try {
    const pod = getPod(courseSlug, podSlug);
    const course = getCourseManifest(courseSlug);

    return (
      <AuthGate>
        <CertificatePageClient
          courseSlug={courseSlug}
          courseTitle={course.title}
          podSlug={pod.slug}
          podTitle={pod.title}
          difficulty={pod.difficulty}
          estimatedHours={pod.estimatedHours}
          notebookCount={pod.notebooks.length}
          notebooks={pod.notebooks}
          hasCaseStudy={!!pod.caseStudy}
        />
      </AuthGate>
    );
  } catch {
    notFound();
  }
}
