import { getPod, getAllPodParams, getCourseManifest } from '@/lib/content';
import { notFound } from 'next/navigation';
import SubscriptionGate from '@/components/auth/SubscriptionGate';
import NotebookPageClient from './NotebookPageClient';

export function generateStaticParams() {
  const podParams = getAllPodParams();
  const params: { courseSlug: string; podSlug: string; id: string }[] = [];
  for (const { courseSlug, podSlug } of podParams) {
    const pod = getPod(courseSlug, podSlug);
    for (const nb of pod.notebooks) {
      params.push({ courseSlug, podSlug, id: String(nb.order) });
    }
  }
  return params;
}

interface PageProps {
  params: Promise<{ courseSlug: string; podSlug: string; id: string }>;
}

export default async function NotebookPage({ params }: PageProps) {
  const { courseSlug, podSlug, id } = await params;
  const order = parseInt(id, 10);

  try {
    const pod = getPod(courseSlug, podSlug);
    const course = getCourseManifest(courseSlug);
    const notebook = pod.notebooks.find((nb) => nb.order === order);
    if (!notebook) notFound();

    const prevNotebook = pod.notebooks.find((nb) => nb.order === order - 1);
    const nextNotebook = pod.notebooks.find((nb) => nb.order === order + 1);

    return (
      <SubscriptionGate podSlug={podSlug}>
        <NotebookPageClient
          courseSlug={courseSlug}
          courseTitle={course.title}
          podSlug={podSlug}
          podTitle={pod.title}
          notebook={notebook}
          prevOrder={prevNotebook?.order}
          nextOrder={nextNotebook?.order}
          totalNotebooks={pod.notebooks.length}
          notebooks={pod.notebooks}
          hasCaseStudy={!!pod.caseStudy}
        />
      </SubscriptionGate>
    );
  } catch {
    notFound();
  }
}
