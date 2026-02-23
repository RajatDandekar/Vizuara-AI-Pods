import { getCourseManifest, getCourseSlugs, getCatalog, getLivePods } from '@/lib/content';
import { notFound } from 'next/navigation';
import CourseOverviewClient from './CourseOverviewClient';

export function generateStaticParams() {
  return getCourseSlugs().map((courseSlug) => ({ courseSlug }));
}

interface PageProps {
  params: Promise<{ courseSlug: string }>;
}

export default async function CourseOverviewPage({ params }: PageProps) {
  const { courseSlug } = await params;

  try {
    const manifest = getCourseManifest(courseSlug);
    const catalog = getCatalog();
    const catalogEntry = catalog.find((c) => c.slug === courseSlug);
    const livePods = getLivePods(courseSlug);

    return (
      <CourseOverviewClient
        courseSlug={manifest.slug}
        title={manifest.title}
        description={manifest.description}
        difficulty={manifest.difficulty}
        estimatedHours={manifest.estimatedHours}
        tags={manifest.tags}
        allPods={manifest.pods}
        livePods={livePods}
        status={catalogEntry?.status}
      />
    );
  } catch {
    notFound();
  }
}
