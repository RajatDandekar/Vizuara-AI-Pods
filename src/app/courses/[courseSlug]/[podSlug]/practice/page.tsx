import { getPod, getAllPodParams, getCourseManifest } from '@/lib/content';
import { notFound, redirect } from 'next/navigation';
import SubscriptionGate from '@/components/auth/SubscriptionGate';
import PracticeHubClient from './PracticeHubClient';

export function generateStaticParams() {
  return getAllPodParams().map(({ courseSlug, podSlug }) => ({ courseSlug, podSlug }));
}

interface PageProps {
  params: Promise<{ courseSlug: string; podSlug: string }>;
}

export default async function PracticePage({ params }: PageProps) {
  const { courseSlug, podSlug } = await params;

  try {
    const pod = getPod(courseSlug, podSlug);
    const course = getCourseManifest(courseSlug);

    if (pod.notebooks.length === 0) {
      redirect(`/courses/${courseSlug}/${podSlug}`);
    }

    return (
      <SubscriptionGate podSlug={podSlug}>
        <PracticeHubClient
          courseSlug={courseSlug}
          courseTitle={course.title}
          podSlug={pod.slug}
          title={pod.title}
          notebooks={pod.notebooks}
        />
      </SubscriptionGate>
    );
  } catch {
    notFound();
  }
}
