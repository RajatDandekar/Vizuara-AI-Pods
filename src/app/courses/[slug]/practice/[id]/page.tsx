import { getCourse, getCourseSlugs } from '@/lib/content';
import { notFound } from 'next/navigation';
import AuthGate from '@/components/auth/AuthGate';
import NotebookPageClient from './NotebookPageClient';

export function generateStaticParams() {
  const slugs = getCourseSlugs();
  const params: { slug: string; id: string }[] = [];
  for (const slug of slugs) {
    const course = getCourse(slug);
    for (const nb of course.notebooks) {
      params.push({ slug, id: String(nb.order) });
    }
  }
  return params;
}

interface PageProps {
  params: Promise<{ slug: string; id: string }>;
}

export default async function NotebookPage({ params }: PageProps) {
  const { slug, id } = await params;
  const order = parseInt(id, 10);

  try {
    const course = getCourse(slug);
    const notebook = course.notebooks.find((nb) => nb.order === order);
    if (!notebook) notFound();

    const prevNotebook = course.notebooks.find((nb) => nb.order === order - 1);
    const nextNotebook = course.notebooks.find((nb) => nb.order === order + 1);

    return (
      <AuthGate>
        <NotebookPageClient
          courseSlug={course.slug}
          courseTitle={course.title}
          notebook={notebook}
          prevOrder={prevNotebook?.order}
          nextOrder={nextNotebook?.order}
          totalNotebooks={course.notebooks.length}
          notebooks={course.notebooks}
          hasCaseStudy={!!course.caseStudy}
        />
      </AuthGate>
    );
  } catch {
    notFound();
  }
}
