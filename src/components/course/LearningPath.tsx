'use client';

import type { NotebookMeta, PodProgress } from '@/types/course';

interface LearningPathProps {
  notebooks: NotebookMeta[];
  progress: PodProgress;
  hasCaseStudy?: boolean;
}

export default function LearningPath({ notebooks, progress, hasCaseStudy }: LearningPathProps) {
  const articleDone = progress.articleRead;
  const allNotebooksDone = notebooks.length > 0 && progress.completedNotebooks.length === notebooks.length;

  function getNotebookStatus(nb: NotebookMeta): 'completed' | 'current' | 'upcoming' {
    if (progress.completedNotebooks.includes(nb.slug)) return 'completed';
    if (!articleDone) return 'upcoming';
    const firstIncomplete = notebooks.find(
      (n) => !progress.completedNotebooks.includes(n.slug)
    );
    if (firstIncomplete && firstIncomplete.slug === nb.slug) return 'current';
    return 'upcoming';
  }

  const caseStudyDone = progress.caseStudyComplete;
  const caseStudyStatus: 'completed' | 'current' | 'upcoming' =
    caseStudyDone ? 'completed' :
    articleDone && allNotebooksDone ? 'current' : 'upcoming';

  const allContentComplete = articleDone && allNotebooksDone && (!hasCaseStudy || caseStudyDone);
  const certificateStatus: 'current' | 'upcoming' =
    allContentComplete ? 'current' : 'upcoming';

  return (
    <div className="w-full overflow-x-auto pb-2">
      <div className="flex items-center gap-0 min-w-max px-2">
        {/* Article node */}
        <div className="flex items-center">
          <div className="flex flex-col items-center">
            <div
              className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-semibold transition-all ${
                articleDone
                  ? 'bg-accent-green text-white'
                  : !articleDone && progress.completedNotebooks.length === 0
                    ? 'bg-accent-blue text-white ring-4 ring-accent-blue/20'
                    : 'bg-gray-200 text-text-muted'
              }`}
            >
              {articleDone ? (
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                </svg>
              ) : (
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              )}
            </div>
            <span className="text-xs text-text-muted mt-1.5 whitespace-nowrap">Article</span>
          </div>
        </div>

        {/* Connector + Notebook nodes */}
        {notebooks.map((nb) => {
          const status = getNotebookStatus(nb);
          return (
            <div key={nb.slug} className="flex items-center">
              <div
                className={`w-8 sm:w-12 h-0.5 ${
                  status === 'completed' ? 'bg-accent-green' :
                  status === 'current' ? 'bg-accent-blue' :
                  'bg-gray-200'
                }`}
              />
              <div className="flex flex-col items-center">
                <div
                  className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-semibold transition-all ${
                    status === 'completed'
                      ? 'bg-accent-green text-white'
                      : status === 'current'
                        ? 'bg-accent-blue text-white ring-4 ring-accent-blue/20'
                        : 'bg-gray-200 text-text-muted'
                  }`}
                >
                  {status === 'completed' ? (
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                    </svg>
                  ) : (
                    nb.order
                  )}
                </div>
                <span className="text-xs text-text-muted mt-1.5 max-w-[80px] text-center truncate whitespace-nowrap">
                  {nb.title.length > 12 ? `Notebook ${nb.order}` : nb.title}
                </span>
              </div>
            </div>
          );
        })}

        {/* Case Study node */}
        {hasCaseStudy && (
          <div className="flex items-center">
            <div
              className={`w-8 sm:w-12 h-0.5 ${
                caseStudyStatus === 'completed' ? 'bg-accent-green' :
                caseStudyStatus === 'current' ? 'bg-accent-amber' : 'bg-gray-200'
              }`}
            />
            <div className="flex flex-col items-center">
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-semibold transition-all ${
                  caseStudyStatus === 'completed'
                    ? 'bg-accent-green text-white'
                    : caseStudyStatus === 'current'
                      ? 'bg-accent-amber text-white ring-4 ring-accent-amber/20'
                      : 'bg-gray-200 text-text-muted'
                }`}
              >
                {caseStudyStatus === 'completed' ? (
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                  </svg>
                ) : (
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M20.25 14.15v4.25c0 1.094-.787 2.036-1.872 2.18-2.087.277-4.216.42-6.378.42s-4.291-.143-6.378-.42c-1.085-.144-1.872-1.086-1.872-2.18v-4.25m16.5 0a2.18 2.18 0 00.75-1.661V8.706c0-1.081-.768-2.015-1.837-2.175a48.114 48.114 0 00-3.413-.387m4.5 8.006c-.194.165-.42.295-.673.38A23.978 23.978 0 0112 15.75c-2.648 0-5.195-.429-7.577-1.22a2.016 2.016 0 01-.673-.38m0 0A2.18 2.18 0 013 12.489V8.706c0-1.081.768-2.015 1.837-2.175a48.111 48.111 0 013.413-.387m7.5 0V5.25A2.25 2.25 0 0013.5 3h-3a2.25 2.25 0 00-2.25 2.25v.894m7.5 0a48.667 48.667 0 00-7.5 0M12 12.75h.008v.008H12v-.008z" />
                  </svg>
                )}
              </div>
              <span className="text-xs text-text-muted mt-1.5 whitespace-nowrap">Case Study</span>
            </div>
          </div>
        )}

        {/* Certificate node */}
        <div className="flex items-center">
          <div
            className={`w-8 sm:w-12 h-0.5 ${
              certificateStatus === 'current' ? 'bg-accent-amber' : 'bg-gray-200'
            }`}
          />
          <div className="flex flex-col items-center">
            <div
              className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-semibold transition-all ${
                certificateStatus === 'current'
                  ? 'bg-accent-amber text-white ring-4 ring-accent-amber/20'
                  : 'bg-gray-200 text-text-muted'
              }`}
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 18.75h-9m9 0a3 3 0 013 3h-15a3 3 0 013-3m9 0v-3.375c0-.621-.503-1.125-1.125-1.125h-.871M7.5 18.75v-3.375c0-.621.504-1.125 1.125-1.125h.872m5.007 0H9.497m5.007 0a7.454 7.454 0 01-.982-3.172M9.497 14.25a7.454 7.454 0 00.981-3.172M5.25 4.236c-.982.143-1.954.317-2.916.52A6.003 6.003 0 007.73 9.728M5.25 4.236V4.5c0 2.108.966 3.99 2.48 5.228M5.25 4.236V2.721C7.456 2.41 9.71 2.25 12 2.25c2.291 0 4.545.16 6.75.47v1.516M18.75 4.236c.982.143 1.954.317 2.916.52A6.003 6.003 0 0016.27 9.728M18.75 4.236V4.5c0 2.108-.966 3.99-2.48 5.228m0 0a6.003 6.003 0 01-5.54 0" />
              </svg>
            </div>
            <span className="text-xs text-text-muted mt-1.5 whitespace-nowrap">Certificate</span>
          </div>
        </div>
      </div>
    </div>
  );
}
