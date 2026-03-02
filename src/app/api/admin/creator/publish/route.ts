import { NextRequest, NextResponse } from 'next/server';
import { getAdminUser } from '@/lib/admin';
import { getProject, updateProject, updateStepStatus, downloadArtifact } from '@/lib/creator-projects';
import { commitAndPush, getRepoFileContent, type FileToCommit } from '@/lib/github-publish';

export const dynamic = 'force-dynamic';
export const maxDuration = 300;

export async function POST(request: NextRequest) {
  const admin = await getAdminUser();
  if (!admin) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  const { projectId } = await request.json();
  const project = await getProject(projectId);
  if (!project) return NextResponse.json({ error: 'Project not found' }, { status: 404 });

  const encoder = new TextEncoder();

  const readableStream = new ReadableStream({
    async start(controller) {
      const sendProgress = (step: string, progress: number) => {
        controller.enqueue(
          encoder.encode(
            `data: ${JSON.stringify({ type: 'progress', step, message: step, progress })}\n\n`
          )
        );
      };

      try {
        const { courseSlug, podSlug } = project;
        const filesToCommit: FileToCommit[] = [];

        // Step 1: Prepare article
        sendProgress('Preparing article...', 10);
        if (project.articleDraft) {
          filesToCommit.push({
            path: `content/courses/${courseSlug}/pods/${podSlug}/article.md`,
            content: project.articleDraft,
          });
        }

        // Step 2: Download and prepare figures
        sendProgress('Preparing figures...', 20);
        const figureUrls: Record<string, string> = {};
        for (const figure of project.figures) {
          if (figure.storagePath) {
            try {
              const figureBuffer = await downloadArtifact(
                projectId,
                `figures/${figure.id}.png`
              );
              filesToCommit.push({
                path: `public/courses/${courseSlug}/pods/${podSlug}/figures/${figure.id}.png`,
                content: figureBuffer,
              });
              figureUrls[`figures/${figure.id}.png`] =
                figure.driveUrl ||
                `/courses/${courseSlug}/pods/${podSlug}/figures/${figure.id}.png`;
            } catch {
              // Figure not in storage, skip
            }
          }
        }

        // Step 3: Download and prepare notebooks
        sendProgress('Preparing notebooks...', 35);
        const notebookMetas = [];
        for (let i = 0; i < project.notebooks.length; i++) {
          const nb = project.notebooks[i];
          if (nb.ipynbStoragePath) {
            try {
              const ipynbBuffer = await downloadArtifact(
                projectId,
                `notebooks/${nb.id}.ipynb`
              );
              filesToCommit.push({
                path: `public/notebooks/${courseSlug}/${podSlug}/${nb.id}.ipynb`,
                content: ipynbBuffer,
              });
            } catch {
              // Notebook not in storage, skip
            }
          }

          notebookMetas.push({
            title: nb.title,
            slug: nb.id,
            objective: nb.objective,
            colabUrl: nb.colabUrl || '',
            downloadPath: `/notebooks/${courseSlug}/${podSlug}/${nb.id}.ipynb`,
            hasNarration: !!nb.colabUrl,
            estimatedMinutes: 25,
            todoCount: 4,
            order: i + 1,
          });
        }

        // Step 4: Prepare case study
        sendProgress('Preparing case study...', 50);
        let caseStudyMeta = undefined;
        if (project.caseStudy?.content) {
          filesToCommit.push({
            path: `public/case-studies/${courseSlug}/${podSlug}/case_study.md`,
            content: project.caseStudy.content,
          });

          caseStudyMeta = {
            title: `${project.podTitle} Case Study`,
            subtitle: `Real-world application of ${project.concept}`,
            company: 'Vizuara Labs',
            industry: 'AI/ML',
            description: `Case study applying ${project.concept} concepts`,
            pdfPath: project.caseStudy.pdfStoragePath
              ? `/case-studies/${courseSlug}/${podSlug}/case_study.pdf`
              : '',
            colabUrl: '',
            notebookPath: '',
          };

          // Download and include PDF if it exists
          if (project.caseStudy.pdfStoragePath) {
            try {
              const pdfBuffer = await downloadArtifact(
                projectId,
                'case-study/case_study.pdf'
              );
              filesToCommit.push({
                path: `public/case-studies/${courseSlug}/${podSlug}/case_study.pdf`,
                content: pdfBuffer,
              });
            } catch {
              // PDF not in storage, skip
            }
          }
        }

        // Step 5: Build pod.json
        sendProgress('Building pod.json...', 60);
        const podManifest = {
          title: project.podTitle,
          slug: podSlug,
          courseSlug,
          order: 99,
          description: project.concept,
          difficulty: 'intermediate' as const,
          estimatedHours: Math.ceil(notebookMetas.length * 0.5),
          thumbnail: project.figures.length > 0
            ? `/courses/${courseSlug}/pods/${podSlug}/figures/figure_1.png`
            : undefined,
          prerequisites: [],
          tags: [],
          article: {
            figureUrls,
          },
          notebooks: notebookMetas,
          caseStudy: caseStudyMeta,
          curator: {
            name: admin.fullName || 'Vizuara Team',
          },
        };

        filesToCommit.push({
          path: `content/courses/${courseSlug}/pods/${podSlug}/pod.json`,
          content: JSON.stringify(podManifest, null, 2),
        });

        // Step 6: Update course.json
        sendProgress('Updating course.json...', 70);
        const courseJsonPath = `content/courses/${courseSlug}/course.json`;
        const courseRaw = await getRepoFileContent(courseJsonPath);
        if (courseRaw) {
          const course = JSON.parse(courseRaw);
          if (!course.pods?.find((p: { slug: string }) => p.slug === podSlug)) {
            course.pods = course.pods || [];
            course.pods.push({
              slug: podSlug,
              title: project.podTitle,
              description: project.concept,
              order: course.pods.length + 1,
              notebookCount: notebookMetas.length,
              estimatedHours: podManifest.estimatedHours,
              hasCaseStudy: !!caseStudyMeta,
              thumbnail: podManifest.thumbnail,
            });

            filesToCommit.push({
              path: courseJsonPath,
              content: JSON.stringify(course, null, 2),
            });
          }
        }

        // Step 7: Update catalog.json
        sendProgress('Updating catalog.json...', 80);
        const catalogPath = 'content/courses/catalog.json';
        const catalogRaw = await getRepoFileContent(catalogPath);
        if (catalogRaw) {
          const catalog = JSON.parse(catalogRaw);
          const courseEntry = catalog.courses?.find(
            (c: { slug: string }) => c.slug === courseSlug
          );
          if (courseEntry) {
            // Only increment if this pod isn't already counted
            const courseJsonContent = filesToCommit.find(f => f.path === courseJsonPath);
            if (courseJsonContent) {
              // We added a new pod, so increment
              courseEntry.podCount = (courseEntry.podCount || 0) + 1;
              courseEntry.totalNotebooks =
                (courseEntry.totalNotebooks || 0) + notebookMetas.length;
            }

            if (!courseEntry.thumbnail && podManifest.thumbnail) {
              courseEntry.thumbnail = podManifest.thumbnail;
            }

            filesToCommit.push({
              path: catalogPath,
              content: JSON.stringify(catalog, null, 2),
            });
          }
        }

        // Step 8: Create atomic commit via GitHub API
        sendProgress('Pushing to GitHub...', 90);
        const commitMessage = `Add pod: ${project.podTitle} (${courseSlug}/${podSlug})`;
        const { sha, url } = await commitAndPush(filesToCommit, commitMessage);

        // Step 9: Mark as published
        sendProgress('Finalizing...', 95);
        await updateProject(projectId, {
          publishedAt: new Date().toISOString(),
          currentStep: 'publish',
        });
        await updateStepStatus(projectId, 'publish', 'complete');

        sendProgress('Published successfully!', 100);
        controller.enqueue(
          encoder.encode(
            `data: ${JSON.stringify({ type: 'done', commitSha: sha, commitUrl: url })}\n\n`
          )
        );
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Publish failed';
        console.error('[publish error]', err);
        controller.enqueue(
          encoder.encode(`data: ${JSON.stringify({ type: 'error', message })}\n\n`)
        );
      }

      controller.close();
    },
  });

  return new Response(readableStream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive',
    },
  });
}
