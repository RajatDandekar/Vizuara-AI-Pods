import { getCourse, getCourseSlugs } from '@/lib/content';
import { extractNotebookContext, extractNotebookTOC } from '@/lib/notebook-context';
import { notFound } from 'next/navigation';
import path from 'path';
import fs from 'fs';
import AuthGate from '@/components/auth/AuthGate';
import AssistantPageClient from './AssistantPageClient';

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

export default async function AssistantPage({ params }: PageProps) {
  const { slug, id } = await params;
  const order = parseInt(id, 10);

  try {
    const course = getCourse(slug);
    const notebook = course.notebooks.find((nb) => nb.order === order);
    if (!notebook) notFound();

    // Find the .ipynb file on disk
    const nbDir = path.join(process.cwd(), 'public', 'notebooks', slug);
    const files = fs.existsSync(nbDir) ? fs.readdirSync(nbDir) : [];
    // Notebooks are named like 01_topic.ipynb, 02_topic.ipynb — match by order prefix
    const orderPrefix = String(order).padStart(2, '0');
    const nbFile = files.find((f) => f.startsWith(orderPrefix) && f.endsWith('.ipynb'));

    let notebookContext = '';
    let toc: string[] = [];
    if (nbFile) {
      const fullPath = path.join(nbDir, nbFile);
      notebookContext = extractNotebookContext(fullPath);
      toc = extractNotebookTOC(fullPath);
    }

    return (
      <AuthGate>
        <AssistantPageClient
          courseSlug={course.slug}
          courseTitle={course.title}
          notebook={notebook}
          notebooks={course.notebooks}
          hasCaseStudy={!!course.caseStudy}
          notebookContext={notebookContext}
          toc={toc}
        />
      </AuthGate>
    );
  } catch {
    notFound();
  }
}
