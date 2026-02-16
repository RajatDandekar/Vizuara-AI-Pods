import { getCourse, getCourseSlugs } from '@/lib/content';
import { notFound, redirect } from 'next/navigation';
import AuthGate from '@/components/auth/AuthGate';
import PracticeHubClient from './PracticeHubClient';

export function generateStaticParams() {
  return getCourseSlugs().map((slug) => ({ slug }));
}

interface PageProps {
  params: Promise<{ slug: string }>;
}

export default async function PracticePage({ params }: PageProps) {
  const { slug } = await params;

  try {
    const course = getCourse(slug);

    // Article-only courses have no practice section
    if (course.notebooks.length === 0) {
      redirect(`/courses/${slug}`);
    }

    return (
      <AuthGate>
        <PracticeHubClient
          slug={course.slug}
          title={course.title}
          notebooks={course.notebooks}
        />
      </AuthGate>
    );
  } catch {
    notFound();
  }
}
