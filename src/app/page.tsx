import { getCatalog, getFullCatalog, getLivePods, getCourseManifest } from '@/lib/content';
import type { PodCard } from '@/types/course';
import HomeClient from './HomeClient';

export default function HomePage() {
  const allCourses = getCatalog();
  const fullCatalog = getFullCatalog();
  const draftCourses = fullCatalog.filter((c) => c.status === 'draft');

  // Build maps of courseSlug → livePods and courseSlug → allPods
  const coursePods: Record<string, PodCard[]> = {};
  const courseAllPods: Record<string, PodCard[]> = {};

  for (const course of fullCatalog) {
    try {
      coursePods[course.slug] = getLivePods(course.slug);
    } catch {
      coursePods[course.slug] = [];
    }
    try {
      courseAllPods[course.slug] = getCourseManifest(course.slug).pods;
    } catch {
      courseAllPods[course.slug] = [];
    }
  }

  return (
    <HomeClient
      courses={allCourses}
      coursePods={coursePods}
      courseAllPods={courseAllPods}
      draftCourses={draftCourses}
    />
  );
}
