import { getCourse, getCourseSlugs, getCatalog } from '@/lib/content';
import { notFound } from 'next/navigation';
import AuthGate from '@/components/auth/AuthGate';
import CourseOverviewClient from './CourseOverviewClient';
import UpcomingCourseClient from './UpcomingCourseClient';

export function generateStaticParams() {
  return getCourseSlugs().map((slug) => ({ slug }));
}

interface PageProps {
  params: Promise<{ slug: string }>;
}

export default async function CourseOverviewPage({ params }: PageProps) {
  const { slug } = await params;

  // Check if this is an upcoming course
  const catalog = getCatalog();
  const catalogEntry = catalog.find((c) => c.slug === slug);

  if (catalogEntry?.status === 'upcoming') {
    return (
      <AuthGate>
        <UpcomingCourseClient course={catalogEntry} />
      </AuthGate>
    );
  }

  try {
    const course = getCourse(slug);
    return (
      <AuthGate>
        <CourseOverviewClient
          slug={course.slug}
          title={course.title}
          description={course.description}
          difficulty={course.difficulty}
          estimatedHours={course.estimatedHours}
          tags={course.tags}
          notebooks={course.notebooks}
          caseStudy={course.caseStudy}
          curator={course.curator}
        />
      </AuthGate>
    );
  } catch {
    notFound();
  }
}
