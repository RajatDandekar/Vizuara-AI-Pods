import { getPod, getAllPodParams, getCourseManifest } from '@/lib/content';
import { notFound } from 'next/navigation';
import AuthGate from '@/components/auth/AuthGate';
import PodOverviewClient from './PodOverviewClient';

export function generateStaticParams() {
  return getAllPodParams().map(({ courseSlug, podSlug }) => ({ courseSlug, podSlug }));
}

interface PageProps {
  params: Promise<{ courseSlug: string; podSlug: string }>;
}

export default async function PodOverviewPage({ params }: PageProps) {
  const { courseSlug, podSlug } = await params;

  try {
    const pod = getPod(courseSlug, podSlug);
    const course = getCourseManifest(courseSlug);

    return (
      <AuthGate>
        <PodOverviewClient
          courseSlug={courseSlug}
          courseTitle={course.title}
          podSlug={pod.slug}
          title={pod.title}
          description={pod.description}
          difficulty={pod.difficulty}
          estimatedHours={pod.estimatedHours}
          tags={pod.tags}
          notebooks={pod.notebooks}
          caseStudy={pod.caseStudy}
          curator={pod.curator}
        />
      </AuthGate>
    );
  } catch {
    notFound();
  }
}
