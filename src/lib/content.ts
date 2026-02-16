import fs from 'fs';
import path from 'path';
import type { CourseManifest, CourseCard, CaseStudySection } from '@/types/course';

const CONTENT_DIR = path.join(process.cwd(), 'content', 'courses');

export function getCatalog(): CourseCard[] {
  const catalogPath = path.join(CONTENT_DIR, 'catalog.json');
  const raw = fs.readFileSync(catalogPath, 'utf-8');
  const data = JSON.parse(raw) as { courses: CourseCard[] };
  return data.courses.filter((c) => c.status !== 'draft');
}

export function getCourse(slug: string): CourseManifest & { articleContent: string } {
  const courseDir = path.join(CONTENT_DIR, slug);
  const manifestPath = path.join(courseDir, 'course.json');
  const articlePath = path.join(courseDir, 'article.md');

  const manifest = JSON.parse(
    fs.readFileSync(manifestPath, 'utf-8')
  ) as CourseManifest;

  const articleContent = fs.readFileSync(articlePath, 'utf-8');

  return { ...manifest, articleContent };
}

export function getCourseSlugs(): string[] {
  const catalogPath = path.join(CONTENT_DIR, 'catalog.json');
  const raw = fs.readFileSync(catalogPath, 'utf-8');
  const data = JSON.parse(raw) as { courses: CourseCard[] };
  // Only return slugs that have a course.json and are not draft
  return data.courses
    .filter((c) => {
      if (c.status === 'draft') return false;
      const manifestPath = path.join(CONTENT_DIR, c.slug, 'course.json');
      return fs.existsSync(manifestPath);
    })
    .map((c) => c.slug);
}

export function getCaseStudyContent(slug: string): string | null {
  const mdPath = path.join(CONTENT_DIR, slug, 'case_study.md');
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
      // Flush previous section
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

  // Flush last section
  if (currentLines.length > 0) {
    const content = currentLines.join('\n').trim();
    if (content) {
      sections.push({ id: currentId, title: currentTitle, content });
    }
  }

  // Process sections: replace Section 3, remove Section 5
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
 * them as LaTeX delimiters. Targets patterns like $23, $180M, $5.50, $0.80/hr.
 * Leaves `$$...$$` (block math) and `$...$` (inline math with non-digit start)
 * untouched.
 */
function escapeCurrencyDollars(md: string): string {
  // Match $ followed by a digit (currency pattern), but NOT $$ (block math)
  // and NOT already escaped \$
  return md.replace(/(?<![\\$])\$(?=\d)/g, '\\$');
}
