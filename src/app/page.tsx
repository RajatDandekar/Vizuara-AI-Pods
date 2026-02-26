import { getCatalog, getFullCatalog, getLivePods, getCourseManifest, getPod } from '@/lib/content';
import { FREE_POD_SPECS } from '@/lib/constants';
import type { PodCard, FreePodShowcase } from '@/types/course';
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

  // Load free pod data for the guest showcase
  const freePods: FreePodShowcase[] = FREE_POD_SPECS.map(({ courseSlug, podSlug }) => {
    const pod = getPod(courseSlug, podSlug);
    const course = getCourseManifest(courseSlug);
    return {
      courseSlug,
      courseTitle: course.title,
      podSlug,
      title: pod.title,
      description: pod.description,
      difficulty: pod.difficulty,
      estimatedHours: pod.estimatedHours,
      notebooks: pod.notebooks,
      caseStudy: pod.caseStudy,
      thumbnail: pod.thumbnail,
      tags: pod.tags,
    };
  });

  return (
    <HomeClient
      courses={allCourses}
      coursePods={coursePods}
      courseAllPods={courseAllPods}
      draftCourses={draftCourses}
      freePods={freePods}
    />
  );
}
