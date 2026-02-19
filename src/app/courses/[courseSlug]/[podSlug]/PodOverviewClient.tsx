'use client';

import { useEffect, useState, useCallback } from 'react';
import Link from 'next/link';
import FadeIn from '@/components/animations/FadeIn';
import Breadcrumb from '@/components/navigation/Breadcrumb';
import CourseHeader from '@/components/course/CourseHeader';
import LearningPath from '@/components/course/LearningPath';
import CoursePhase from '@/components/course/CoursePhase';
import Button from '@/components/ui/Button';
import Badge from '@/components/ui/Badge';
import CourseProgressBar from '@/components/course/CourseProgressBar';
import CuratorVideo from '@/components/course/CuratorVideo';
import type { NotebookMeta, CaseStudyMeta, CuratorInfo, PodProgress } from '@/types/course';
import { getPodProgress, onProgressChange } from '@/lib/progress';

interface Props {
  courseSlug: string;
  courseTitle: string;
  podSlug: string;
  title: string;
  description: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  estimatedHours: number;
  tags: string[];
  notebooks: NotebookMeta[];
  caseStudy?: CaseStudyMeta;
  curator?: CuratorInfo;
}

export default function PodOverviewClient({
  courseSlug,
  courseTitle,
  podSlug,
  title,
  description,
  difficulty,
  estimatedHours,
  tags,
  notebooks,
  caseStudy,
  curator,
}: Props) {
  const [progress, setProgress] = useState<PodProgress>({
    articleRead: false,
    completedNotebooks: [],
    caseStudyComplete: false,
    lastVisited: '',
  });

  useEffect(() => {
    setProgress(getPodProgress(courseSlug, podSlug));
    return onProgressChange(() => setProgress(getPodProgress(courseSlug, podSlug)));
  }, [courseSlug, podSlug]);

  const hasStarted = progress.articleRead || progress.completedNotebooks.length > 0;
  const allNotebooksDone = notebooks.length > 0 && progress.completedNotebooks.length === notebooks.length;
  const caseStudyDone = progress.caseStudyComplete;
  const articleAndNotebooksDone = progress.articleRead && (notebooks.length === 0 || allNotebooksDone);
  const allContentComplete = articleAndNotebooksDone && (!caseStudy || caseStudyDone);

  const articleStatus = progress.articleRead ? 'completed' as const : 'current' as const;
  const notebookStatus = !progress.articleRead
    ? 'locked' as const
    : allNotebooksDone
      ? 'completed' as const
      : 'current' as const;
  const caseStudyStatus = caseStudyDone
    ? 'completed' as const
    : articleAndNotebooksDone
      ? 'current' as const
      : 'locked' as const;
  const certificateStatus = allContentComplete ? 'current' as const : 'locked' as const;

  const getInitialPhase = useCallback(() => {
    if (!progress.articleRead) return 1;
    if (notebooks.length > 0 && !allNotebooksDone) return 2;
    if (caseStudy && articleAndNotebooksDone && !caseStudyDone) return 3;
    if (allContentComplete) return 4;
    return 1;
  }, [progress.articleRead, notebooks.length, allNotebooksDone, caseStudy, articleAndNotebooksDone, caseStudyDone, allContentComplete]);

  const [expandedPhase, setExpandedPhase] = useState<number | null>(null);

  useEffect(() => {
    setExpandedPhase(getInitialPhase());
  }, [getInitialPhase]);

  const togglePhase = (phase: number) => {
    setExpandedPhase((prev) => (prev === phase ? null : phase));
  };

  const basePath = `/courses/${courseSlug}/${podSlug}`;

  let continueHref = `${basePath}/article`;
  let continueLabel = 'Start Learning';

  if (allContentComplete) {
    continueHref = `${basePath}/certificate`;
    continueLabel = 'Get Your Certificate';
  } else if (progress.articleRead && notebooks.length === 0 && !caseStudy) {
    continueHref = `${basePath}/certificate`;
    continueLabel = 'Get Your Certificate';
  } else if (progress.articleRead) {
    const firstIncomplete = notebooks.find(
      (nb) => !progress.completedNotebooks.includes(nb.slug)
    );
    if (firstIncomplete) {
      continueHref = `${basePath}/practice/${firstIncomplete.order}`;
      continueLabel = 'Continue Learning';
    } else if (caseStudy && !caseStudyDone) {
      continueHref = `${basePath}/case-study`;
      continueLabel = 'Start Case Study';
    } else {
      continueHref = `${basePath}/certificate`;
      continueLabel = 'Get Your Certificate';
    }
  } else if (hasStarted) {
    continueLabel = 'Continue Reading';
  }

  const completedNbCount = progress.completedNotebooks.length;
  const totalMinutes = notebooks.reduce((sum, nb) => sum + nb.estimatedMinutes, 0);

  const activeStep = (() => {
    if (!progress.articleRead) return 'article';
    const firstIncompleteNb = notebooks.find(
      (nb) => !progress.completedNotebooks.includes(nb.slug)
    );
    if (firstIncompleteNb) return `nb-${firstIncompleteNb.order}`;
    if (caseStudy && !caseStudyDone) return 'case-study';
    return 'certificate';
  })();

  return (
    <>
      {hasStarted && (
        <CourseProgressBar
          courseSlug={courseSlug}
          podSlug={podSlug}
          courseTitle={title}
          notebooks={notebooks}
          hasCaseStudy={!!caseStudy}
          activeStep={activeStep}
        />
      )}
    <div className="max-w-4xl mx-auto px-4 sm:px-6 py-8">
      <FadeIn>
        <Breadcrumb
          items={[
            { label: 'Courses', href: '/' },
            { label: courseTitle, href: `/courses/${courseSlug}` },
            { label: title },
          ]}
        />
      </FadeIn>

      <FadeIn delay={0.1}>
        <CourseHeader
          title={title}
          description={description}
          difficulty={difficulty}
          estimatedHours={estimatedHours}
          notebookCount={notebooks.length}
          tags={tags}
        />
      </FadeIn>

      <FadeIn delay={0.15}>
        <div className="flex items-start gap-3 rounded-xl bg-amber-50 border border-amber-200/60 px-4 py-3 mb-6">
          <svg className="w-5 h-5 text-accent-amber flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
          </svg>
          <p className="text-sm text-amber-800 leading-relaxed">
            <span className="font-semibold">Please note:</span> The curator video below is subject to change and is shared for review purposes only.
          </p>
        </div>
      </FadeIn>

      {curator?.videoUrl && (
        <FadeIn delay={0.18}>
          <CuratorVideo curator={curator} />
        </FadeIn>
      )}

      {notebooks.length > 0 && (
        <FadeIn delay={0.2}>
          <div className="bg-card-bg border border-card-border rounded-2xl p-6 mb-8">
            <h2 className="text-sm font-semibold text-text-muted uppercase tracking-wider mb-4">
              Learning Path
            </h2>
            <LearningPath notebooks={notebooks} progress={progress} hasCaseStudy={!!caseStudy} />
          </div>
        </FadeIn>
      )}

      <FadeIn delay={0.3}>
        <div className="flex items-center gap-4 mb-10">
          <Link href={continueHref}>
            <Button size="lg">
              {continueLabel}
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
              </svg>
            </Button>
          </Link>
          {allContentComplete && (
            <Badge variant="green" size="md">Pod Complete</Badge>
          )}
        </div>
      </FadeIn>

      <FadeIn delay={0.35}>
        <div className="space-y-3">
          <CoursePhase
            phase={1}
            title="Read the Article"
            subtitle="Comprehensive explanation with figures, equations, and code examples"
            status={articleStatus}
            statusLabel={progress.articleRead ? 'Done' : undefined}
            expanded={expandedPhase === 1}
            onToggle={() => togglePhase(1)}
          >
            <Link href={`${basePath}/article`}>
              <div className="bg-gray-50 border border-card-border rounded-xl p-4 hover:shadow-md hover:border-accent-blue/30 transition-all cursor-pointer">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-lg bg-blue-50 text-accent-blue flex items-center justify-center flex-shrink-0">
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                  </div>
                  <div className="flex-1 min-w-0">
                    <h4 className="font-semibold text-foreground text-sm">{title}</h4>
                    <p className="text-xs text-text-secondary">{description}</p>
                  </div>
                  <svg className="w-4 h-4 text-text-muted flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
                  </svg>
                </div>
              </div>
            </Link>
          </CoursePhase>

          {notebooks.length > 0 && (
            <CoursePhase
              phase={2}
              title="Practice with Notebooks"
              subtitle={`${notebooks.length} hands-on Colab notebooks · ~${Math.round(totalMinutes / 60)} hours`}
              status={notebookStatus}
              statusLabel={
                notebookStatus === 'completed'
                  ? 'Done'
                  : notebookStatus === 'current'
                    ? `${completedNbCount}/${notebooks.length}`
                    : undefined
              }
              expanded={expandedPhase === 2}
              onToggle={() => togglePhase(2)}
            >
              <div className="space-y-2">
                {notebooks.map((nb) => {
                  const done = progress.completedNotebooks.includes(nb.slug);
                  const isAccessible = notebookStatus !== 'locked';
                  const content = (
                    <div className={`bg-gray-50 border border-card-border rounded-xl p-4 transition-all ${isAccessible ? 'hover:shadow-md hover:border-accent-blue/30 cursor-pointer' : 'opacity-60'}`}>
                      <div className="flex items-center gap-3">
                        <div
                          className={`w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 text-xs font-semibold ${
                            done
                              ? 'bg-accent-green-light text-accent-green'
                              : 'bg-gray-100 text-text-secondary'
                          }`}
                        >
                          {nb.order}
                        </div>
                        <div className="flex-1 min-w-0">
                          <h4 className="font-medium text-foreground text-sm truncate">{nb.title}</h4>
                        </div>
                        <div className="flex items-center gap-2 flex-shrink-0">
                          <span className="text-xs text-text-muted">{nb.estimatedMinutes} min</span>
                          {done && <Badge variant="green" size="sm">Done</Badge>}
                        </div>
                      </div>
                    </div>
                  );
                  return isAccessible ? (
                    <Link key={nb.slug} href={`${basePath}/practice/${nb.order}`}>
                      {content}
                    </Link>
                  ) : (
                    <div key={nb.slug}>{content}</div>
                  );
                })}
              </div>
            </CoursePhase>
          )}

          {caseStudy && (
            <CoursePhase
              phase={notebooks.length > 0 ? 3 : 2}
              title="Apply Your Knowledge"
              subtitle="Real-world industry case study with implementation notebook"
              status={caseStudyStatus}
              statusLabel={
                caseStudyStatus === 'completed'
                  ? 'Done'
                  : caseStudyStatus === 'locked'
                    ? 'Locked'
                    : undefined
              }
              expanded={expandedPhase === 3}
              onToggle={() => togglePhase(3)}
            >
              {caseStudyStatus === 'locked' ? (
                <div className="text-center py-4 text-sm text-text-muted">
                  <svg className="w-8 h-8 mx-auto mb-2 text-text-muted/50" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 10.5V6.75a4.5 4.5 0 10-9 0v3.75m-.75 11.25h10.5a2.25 2.25 0 002.25-2.25v-6.75a2.25 2.25 0 00-2.25-2.25H6.75a2.25 2.25 0 00-2.25 2.25v6.75a2.25 2.25 0 002.25 2.25z" />
                  </svg>
                  Complete the article and all notebooks to unlock the case study.
                </div>
              ) : (
                <div className="space-y-4">
                  <Link href={`${basePath}/case-study`}>
                    <div className="bg-gradient-to-br from-gray-50 to-amber-50/30 border border-card-border rounded-xl p-5 hover:shadow-md hover:border-amber-300/50 transition-all cursor-pointer">
                      <div className="flex items-center gap-2 mb-2 flex-wrap">
                        {caseStudy.company && (
                          <span className="text-xs font-medium text-accent-amber bg-accent-amber-light px-2 py-0.5 rounded-full">
                            {caseStudy.company}
                          </span>
                        )}
                        {caseStudy.industry && (
                          <span className="text-xs text-text-muted">{caseStudy.industry}</span>
                        )}
                      </div>
                      <h4 className="text-base font-semibold text-foreground mb-1">{caseStudy.title}</h4>
                      {caseStudy.subtitle && (
                        <p className="text-sm text-text-secondary mb-3">{caseStudy.subtitle}</p>
                      )}
                      <p className="text-sm text-text-secondary leading-relaxed mb-4 line-clamp-2">
                        {caseStudy.description}
                      </p>
                    </div>
                  </Link>

                  <div className="flex items-center gap-3 flex-wrap">
                    {caseStudy.colabUrl && (
                      <a href={caseStudy.colabUrl} target="_blank" rel="noopener noreferrer">
                        <Button variant="secondary" size="sm">
                          Open in Colab
                        </Button>
                      </a>
                    )}
                    {caseStudy.notebookPath && (
                      <a href={caseStudy.notebookPath} download>
                        <Button variant="secondary" size="sm">
                          Download Notebook
                        </Button>
                      </a>
                    )}
                    {caseStudy.pdfPath && (
                      <a href={caseStudy.pdfPath} download>
                        <Button variant="secondary" size="sm">
                          Download PDF
                        </Button>
                      </a>
                    )}
                  </div>
                </div>
              )}
            </CoursePhase>
          )}

          <CoursePhase
            phase={caseStudy ? (notebooks.length > 0 ? 4 : 3) : (notebooks.length > 0 ? 3 : 2)}
            title="Get Your Certificate"
            subtitle="Download and share your achievement with the world"
            status={certificateStatus}
            statusLabel={certificateStatus === 'locked' ? 'Locked' : undefined}
            expanded={expandedPhase === 4}
            onToggle={() => togglePhase(4)}
          >
            {certificateStatus === 'locked' ? (
              <div className="text-center py-4 text-sm text-text-muted">
                <svg className="w-8 h-8 mx-auto mb-2 text-text-muted/50" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 10.5V6.75a4.5 4.5 0 10-9 0v3.75m-.75 11.25h10.5a2.25 2.25 0 002.25-2.25v-6.75a2.25 2.25 0 00-2.25-2.25H6.75a2.25 2.25 0 00-2.25 2.25v6.75a2.25 2.25 0 002.25 2.25z" />
                </svg>
                Complete all pod sections to unlock your certificate.
              </div>
            ) : (
              <Link href={`${basePath}/certificate`}>
                <div className="bg-gradient-to-br from-amber-50 to-yellow-50/40 border border-amber-200/60 rounded-xl p-5 hover:shadow-md hover:border-amber-300/60 transition-all cursor-pointer">
                  <div className="flex items-center gap-4">
                    <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-amber-400 to-yellow-500 flex items-center justify-center flex-shrink-0">
                      <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 18.75h-9m9 0a3 3 0 013 3h-15a3 3 0 013-3m9 0v-3.375c0-.621-.503-1.125-1.125-1.125h-.871M7.5 18.75v-3.375c0-.621.504-1.125 1.125-1.125h.872m5.007 0H9.497m5.007 0a7.454 7.454 0 01-.982-3.172M9.497 14.25a7.454 7.454 0 00.981-3.172M5.25 4.236c-.982.143-1.954.317-2.916.52A6.003 6.003 0 007.73 9.728M5.25 4.236V4.5c0 2.108.966 3.99 2.48 5.228M5.25 4.236V2.721C7.456 2.41 9.71 2.25 12 2.25c2.291 0 4.545.16 6.75.47v1.516M18.75 4.236c.982.143 1.954.317 2.916.52A6.003 6.003 0 0016.27 9.728M18.75 4.236V4.5c0 2.108-.966 3.99-2.48 5.228m0 0a6.003 6.003 0 01-5.54 0" />
                      </svg>
                    </div>
                    <div className="flex-1 min-w-0">
                      <h4 className="font-semibold text-foreground text-sm">Certificate of Completion</h4>
                      <p className="text-xs text-text-secondary mt-0.5">
                        Your certificate is ready! Download it and share on LinkedIn & X.
                      </p>
                    </div>
                    <svg className="w-4 h-4 text-text-muted flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
                    </svg>
                  </div>
                </div>
              </Link>
            )}
          </CoursePhase>
        </div>
      </FadeIn>
    </div>
    </>
  );
}
