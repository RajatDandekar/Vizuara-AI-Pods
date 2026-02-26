import { getPod, getAllPodParams, getCourseManifest } from '@/lib/content';
import { extractNotebookContext, extractNotebookTOC } from '@/lib/notebook-context';
import { notFound } from 'next/navigation';
import path from 'path';
import fs from 'fs';
import SubscriptionGate from '@/components/auth/SubscriptionGate';
import AssistantPageClient from './AssistantPageClient';

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

export default async function AssistantPage({ params }: PageProps) {
  const { courseSlug, podSlug, id } = await params;
  const order = parseInt(id, 10);

  try {
    const pod = getPod(courseSlug, podSlug);
    const notebook = pod.notebooks.find((nb) => nb.order === order);
    if (!notebook) notFound();

    // Find the .ipynb file on disk — try new path first, then legacy
    let notebookContext = '';
    let toc: string[] = [];

    const newNbDir = path.join(process.cwd(), 'public', 'notebooks', courseSlug, podSlug);
    const legacyNbDir = path.join(process.cwd(), 'public', 'notebooks', podSlug);
    const nbDir = fs.existsSync(newNbDir) ? newNbDir : legacyNbDir;

    const files = fs.existsSync(nbDir) ? fs.readdirSync(nbDir).filter(f => !fs.statSync(path.join(nbDir, f)).isDirectory()) : [];
    const orderPrefix = String(order).padStart(2, '0');
    const nbFile = files.find((f) => f.startsWith(orderPrefix) && f.endsWith('.ipynb'));

    if (nbFile) {
      const fullPath = path.join(nbDir, nbFile);
      notebookContext = extractNotebookContext(fullPath);
      toc = extractNotebookTOC(fullPath);
    }

    return (
      <SubscriptionGate podSlug={podSlug}>
        <AssistantPageClient
          courseSlug={courseSlug}
          courseTitle={pod.title}
          notebook={notebook}
          notebooks={pod.notebooks}
          hasCaseStudy={!!pod.caseStudy}
          notebookContext={notebookContext}
          toc={toc}
          podSlug={podSlug}
        />
      </SubscriptionGate>
    );
  } catch {
    notFound();
  }
}
