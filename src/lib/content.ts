import fs from 'fs';
import path from 'path';
import type {
  CourseManifest,
  CourseCard,
  PodManifest,
  PodCard,
  CaseStudySection,
} from '@/types/course';

const CONTENT_DIR = path.join(process.cwd(), 'content', 'courses');

// ─── Course-level functions ──────────────────────────────────────────

/** Returns all course-level cards from the catalog (filters out drafts). */
export function getCatalog(): CourseCard[] {
  const catalogPath = path.join(CONTENT_DIR, 'catalog.json');
  const raw = fs.readFileSync(catalogPath, 'utf-8');
  const data = JSON.parse(raw) as { courses: CourseCard[] };
  return data.courses.filter((c) => c.status !== 'draft');
}

/** Returns ALL courses including drafts. */
export function getFullCatalog(): CourseCard[] {
  const catalogPath = path.join(CONTENT_DIR, 'catalog.json');
  const raw = fs.readFileSync(catalogPath, 'utf-8');
  const data = JSON.parse(raw) as { courses: CourseCard[] };
  return data.courses;
}

/** Returns the full course manifest (with pod listing) for a given course slug. */
export function getCourseManifest(courseSlug: string): CourseManifest {
  const manifestPath = path.join(CONTENT_DIR, courseSlug, 'course.json');
  return JSON.parse(fs.readFileSync(manifestPath, 'utf-8')) as CourseManifest;
}

/** Returns course-level slugs that are not drafts and have a course.json with pods. */
export function getCourseSlugs(): string[] {
  const catalog = getCatalog();
  return catalog
    .filter((c) => {
      const manifestPath = path.join(CONTENT_DIR, c.slug, 'course.json');
      if (!fs.existsSync(manifestPath)) return false;
      const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8')) as CourseManifest;
      return manifest.pods && manifest.pods.length > 0;
    })
    .map((c) => c.slug);
}

// ─── Pod-level functions ─────────────────────────────────────────────

/** Reads pod.json + article.md for a specific pod. */
export function getPod(
  courseSlug: string,
  podSlug: string
): PodManifest & { articleContent: string } {
  const podDir = path.join(CONTENT_DIR, courseSlug, 'pods', podSlug);
  const manifestPath = path.join(podDir, 'pod.json');
  const articlePath = path.join(podDir, 'article.md');

  const manifest = JSON.parse(
    fs.readFileSync(manifestPath, 'utf-8')
  ) as PodManifest;

  let articleContent = '';
  if (fs.existsSync(articlePath)) {
    articleContent = fs.readFileSync(articlePath, 'utf-8');
  }

  return { ...manifest, articleContent };
}

/** Returns non-draft pod slugs for a course (those with a pod.json). */
export function getPodSlugs(courseSlug: string): string[] {
  const manifest = getCourseManifest(courseSlug);
  return manifest.pods
    .filter((p: PodCard) => {
      const podJsonPath = path.join(CONTENT_DIR, courseSlug, 'pods', p.slug, 'pod.json');
      return fs.existsSync(podJsonPath);
    })
    .sort((a: PodCard, b: PodCard) => a.order - b.order)
    .map((p: PodCard) => p.slug);
}

/** Returns all {courseSlug, podSlug} pairs for generateStaticParams(). */
export function getAllPodParams(): { courseSlug: string; podSlug: string }[] {
  const courseSlugs = getCourseSlugs();
  const params: { courseSlug: string; podSlug: string }[] = [];

  for (const courseSlug of courseSlugs) {
    const podSlugs = getPodSlugs(courseSlug);
    for (const podSlug of podSlugs) {
      // Only include pods that have an article (i.e., are live)
      const articlePath = path.join(CONTENT_DIR, courseSlug, 'pods', podSlug, 'article.md');
      if (fs.existsSync(articlePath)) {
        params.push({ courseSlug, podSlug });
      }
    }
  }

  return params;
}

/** Returns the list of live pods (those with article.md) for a course. */
export function getLivePods(courseSlug: string): PodCard[] {
  const manifest = getCourseManifest(courseSlug);
  return manifest.pods
    .filter((p: PodCard) => {
      const articlePath = path.join(CONTENT_DIR, courseSlug, 'pods', p.slug, 'article.md');
      return fs.existsSync(articlePath);
    })
    .sort((a: PodCard, b: PodCard) => a.order - b.order);
}

// ─── Case Study ──────────────────────────────────────────────────────

export function getPodCaseStudyContent(courseSlug: string, podSlug: string): string | null {
  const mdPath = path.join(CONTENT_DIR, courseSlug, 'pods', podSlug, 'case_study.md');
  try {
    const raw = fs.readFileSync(mdPath, 'utf-8');
    return escapeCurrencyDollars(raw);
  } catch {
    return null;
  }
}

/**
 * Parse the case study markdown into structured sections for step-by-step
 * display. Splits on `## Section N:` headings, strips Section 3
 * (implementation scaffold — belongs in the Colab notebook) and Section 5
 * (portfolio/resume — removed), and replaces Section 3 with a short pointer.
 */
export function parseCaseStudySections(md: string): CaseStudySection[] {
  const escaped = escapeCurrencyDollars(md);
  const lines = escaped.split('\n');
  const sections: CaseStudySection[] = [];

  let currentId = 'intro';
  let currentTitle = 'Overview';
  let currentLines: string[] = [];

  for (const line of lines) {
    const sectionMatch = line.match(/^## Section (\d+):\s*(.+)/i);
    if (sectionMatch) {
      if (currentLines.length > 0) {
        const content = currentLines.join('\n').trim();
        if (content) {
          sections.push({ id: currentId, title: currentTitle, content });
        }
      }
      const num = sectionMatch[1];
      currentId = `section-${num}`;
      currentTitle = sectionMatch[2].trim();
      currentLines = [];
      continue;
    }

    currentLines.push(line);
  }

  if (currentLines.length > 0) {
    const content = currentLines.join('\n').trim();
    if (content) {
      sections.push({ id: currentId, title: currentTitle, content });
    }
  }

  return sections
    .map((s) => {
      if (s.id === 'section-3') {
        return {
          ...s,
          title: 'Implementation Notebook',
          content:
            'The full implementation — including data acquisition, model design, ' +
            'training loops, evaluation, and error analysis — is provided as a ' +
            'hands-on Google Colab notebook with guided TODO exercises.\n\n' +
            'Use the **Open in Colab** or **Download Notebook** button at the top to get started.',
        };
      }
      return s;
    })
    .filter((s) => s.id !== 'section-5');
}

/**
 * Escape bare `$` signs used for currency so remarkMath doesn't interpret
 * them as LaTeX delimiters.
 */
function escapeCurrencyDollars(md: string): string {
  return md.replace(/(?<![\\$])\$(?=\d)/g, '\\$');
}
