import { getCourseManifest, getCourseSlugs, getLivePods } from '@/lib/content';
import { notFound } from 'next/navigation';
import SubscriptionGate from '@/components/auth/SubscriptionGate';
import CourseCertificateClient from './CourseCertificateClient';

export function generateStaticParams() {
  return getCourseSlugs().map((courseSlug) => ({ courseSlug }));
}

interface PageProps {
  params: Promise<{ courseSlug: string }>;
}

export default async function CourseCertificatePage({ params }: PageProps) {
  const { courseSlug } = await params;

  try {
    const manifest = getCourseManifest(courseSlug);
    const livePods = getLivePods(courseSlug);

    return (
      <SubscriptionGate>
        <CourseCertificateClient
          courseSlug={manifest.slug}
          courseTitle={manifest.title}
          difficulty={manifest.difficulty}
          estimatedHours={manifest.estimatedHours}
          livePods={livePods}
        />
      </SubscriptionGate>
    );
  } catch {
    notFound();
  }
}
