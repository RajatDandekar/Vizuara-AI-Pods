'use client';

import { useEffect, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import FadeIn from '@/components/animations/FadeIn';
import Breadcrumb from '@/components/navigation/Breadcrumb';
import CourseNav from '@/components/navigation/CourseNav';
import CourseProgressBar from '@/components/course/CourseProgressBar';
import MarkdownRenderer from '@/components/ui/MarkdownRenderer';
import Button from '@/components/ui/Button';
import Badge from '@/components/ui/Badge';
import type { CaseStudyMeta, CaseStudySection, NotebookMeta } from '@/types/course';
import {
  isCaseStudyComplete,
  markCaseStudyComplete as markCaseStudyCompleteStorage,
} from '@/lib/progress';

interface Props {
  slug: string;
  courseTitle: string;
  caseStudy: CaseStudyMeta;
  sections: CaseStudySection[];
  notebooks: NotebookMeta[];
}

const STEP_ICONS: Record<string, React.ReactNode> = {
  'section-1': (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 21h19.5m-18-18v18m10.5-18v18m6-13.5V21M6.75 6.75h.75m-.75 3h.75m-.75 3h.75m3-6h.75m-.75 3h.75m-.75 3h.75M6.75 21v-3.375c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125V21M3 3h12m-.75 4.5H21m-3.75 7.5h.008v.008h-.008v-.008zm0 3h.008v.008h-.008v-.008z" />
    </svg>
  ),
  'section-2': (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M4.26 10.147a60.438 60.438 0 00-.491 6.347A48.62 48.62 0 0112 20.904a48.62 48.62 0 018.232-4.41 60.46 60.46 0 00-.491-6.347m-15.482 0a50.636 50.636 0 00-2.658-.813A59.906 59.906 0 0112 3.493a59.903 59.903 0 0110.399 5.84c-.896.248-1.783.52-2.658.814m-15.482 0A50.717 50.717 0 0112 13.489a50.702 50.702 0 017.74-3.342M6.75 15a.75.75 0 100-1.5.75.75 0 000 1.5zm0 0v-3.675A55.378 55.378 0 0112 8.443m-7.007 11.55A5.981 5.981 0 006.75 15.75v-1.5" />
    </svg>
  ),
  'section-3': (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M17.25 6.75L22.5 12l-5.25 5.25m-10.5 0L1.5 12l5.25-5.25m7.5-3l-4.5 16.5" />
    </svg>
  ),
  'section-4': (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 14.25h13.5m-13.5 0a3 3 0 01-3-3m3 3a3 3 0 100 6h13.5a3 3 0 100-6m-16.5-3a3 3 0 013-3h13.5a3 3 0 013 3m-19.5 0a4.5 4.5 0 01.9-2.7L5.737 5.1a3.375 3.375 0 012.7-1.35h7.126c1.062 0 2.062.5 2.7 1.35l2.587 3.45a4.5 4.5 0 01.9 2.7m0 0a3 3 0 01-3 3m0 3h.008v.008h-.008v-.008zm0-6h.008v.008h-.008v-.008zm-3 6h.008v.008h-.008v-.008zm0-6h.008v.008h-.008v-.008z" />
    </svg>
  ),
};

const STEP_LABELS: Record<string, string> = {
  'section-1': 'The Problem',
  'section-2': 'Technical Formulation',
  'section-3': 'Build It',
  'section-4': 'Production Design',
};

export default function CaseStudyPageClient({
  slug,
  courseTitle,
  caseStudy,
  sections,
  notebooks,
}: Props) {
  const [completed, setCompleted] = useState(false);
  const [activeStep, setActiveStep] = useState(0);

  // Filter to only the main sections (section-1 through section-4)
  const steps = sections.filter((s) => s.id.startsWith('section-'));

  useEffect(() => {
    setCompleted(isCaseStudyComplete(slug));
  }, [slug]);

  const handleMarkComplete = () => {
    markCaseStudyCompleteStorage(slug);
    setCompleted(true);
  };

  const currentSection = steps[activeStep];
  const isFirst = activeStep === 0;
  const isLast = activeStep === steps.length - 1;

  return (
    <>
      <CourseProgressBar
        courseSlug={slug}
        courseTitle={courseTitle}
        notebooks={notebooks}
        hasCaseStudy={true}
        activeStep="case-study"
      />
      <div className="max-w-3xl mx-auto px-4 sm:px-6 py-8">
        <FadeIn>
          <Breadcrumb
            items={[
              { label: 'Courses', href: '/' },
              { label: courseTitle, href: `/courses/${slug}` },
              { label: 'Case Study' },
            ]}
          />
        </FadeIn>

        {/* Hero header */}
        <FadeIn delay={0.1}>
          <div className="mb-6">
            <div className="flex items-center gap-2 mb-3">
              <Badge variant="amber" size="md">Industry Case Study</Badge>
              {caseStudy.company && (
                <Badge variant="default" size="sm">{caseStudy.company}</Badge>
              )}
            </div>
            <h1 className="text-2xl sm:text-3xl font-bold text-foreground tracking-tight mb-2">
              {caseStudy.title}
            </h1>
            {caseStudy.subtitle && (
              <p className="text-text-secondary leading-relaxed text-base">
                {caseStudy.subtitle}
              </p>
            )}
          </div>
        </FadeIn>

        {/* Action buttons */}
        <FadeIn delay={0.15}>
          <div className="flex items-center gap-3 mb-6 flex-wrap">
            {caseStudy.colabUrl && (
              <a href={caseStudy.colabUrl} target="_blank" rel="noopener noreferrer">
                <Button size="sm">
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 6H5.25A2.25 2.25 0 003 8.25v10.5A2.25 2.25 0 005.25 21h10.5A2.25 2.25 0 0018 18.75V10.5m-10.5 6L21 3m0 0h-5.25M21 3v5.25" />
                  </svg>
                  Open Notebook in Colab
                </Button>
              </a>
            )}
            {caseStudy.pdfPath && (
              <a href={caseStudy.pdfPath} download>
                <Button variant="secondary" size="sm">
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
                  </svg>
                  Download PDF
                </Button>
              </a>
            )}
          </div>
        </FadeIn>

        {/* Step indicator */}
        <FadeIn delay={0.2}>
          <div className="flex items-center gap-1 mb-8 overflow-x-auto pb-1">
            {steps.map((step, i) => {
              const label = STEP_LABELS[step.id] || step.title;
              const icon = STEP_ICONS[step.id];
              const isActive = i === activeStep;
              const isPast = i < activeStep;

              return (
                <button
                  key={step.id}
                  onClick={() => setActiveStep(i)}
                  className={`
                    flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium
                    transition-all duration-200 cursor-pointer whitespace-nowrap flex-shrink-0
                    ${isActive
                      ? 'bg-accent-blue text-white shadow-sm'
                      : isPast
                        ? 'bg-accent-blue-light text-accent-blue hover:bg-blue-100'
                        : 'bg-gray-50 text-text-secondary hover:bg-gray-100 hover:text-foreground'
                    }
                  `}
                >
                  {isPast && !isActive ? (
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                    </svg>
                  ) : (
                    icon
                  )}
                  <span className="hidden sm:inline">{label}</span>
                  <span className="sm:hidden">{i + 1}</span>
                </button>
              );
            })}
          </div>
        </FadeIn>

        {/* Section content with animation */}
        <AnimatePresence mode="wait">
          {currentSection && (
            <motion.article
              key={currentSection.id}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              transition={{ duration: 0.25, ease: 'easeOut' }}
              className="max-w-none min-h-[300px]"
            >
              <div className="mb-4">
                <p className="text-xs font-semibold text-text-secondary uppercase tracking-wider mb-1">
                  Step {activeStep + 1} of {steps.length}
                </p>
                <h2 className="text-xl font-bold text-foreground">
                  {STEP_LABELS[currentSection.id] || currentSection.title}
                </h2>
              </div>
              <MarkdownRenderer content={currentSection.content} />
            </motion.article>
          )}
        </AnimatePresence>

        {/* Step navigation */}
        <div className="flex items-center justify-between mt-10 pt-6 border-t border-card-border">
          <button
            onClick={() => setActiveStep((s) => Math.max(0, s - 1))}
            disabled={isFirst}
            className={`
              inline-flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium
              transition-all duration-200 cursor-pointer
              ${isFirst
                ? 'text-gray-300 cursor-not-allowed'
                : 'text-text-secondary hover:text-foreground hover:bg-gray-50'
              }
            `}
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 19.5L8.25 12l7.5-7.5" />
            </svg>
            Previous
          </button>

          {!isLast ? (
            <button
              onClick={() => setActiveStep((s) => Math.min(steps.length - 1, s + 1))}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-medium bg-accent-blue text-white hover:bg-blue-700 transition-all duration-200 cursor-pointer"
            >
              Next Step
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
              </svg>
            </button>
          ) : !completed ? (
            <button
              onClick={handleMarkComplete}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-medium bg-accent-green text-white hover:bg-green-700 transition-all duration-200 cursor-pointer"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
              </svg>
              Mark Case Study as Complete
            </button>
          ) : (
            <Badge variant="green" size="md">Case Study Complete</Badge>
          )}
        </div>

        <FadeIn delay={0.3}>
          <div className="mt-6">
            <CourseNav
              prevHref={`/courses/${slug}`}
              prevLabel="Course Overview"
              nextHref={completed ? `/courses/${slug}/certificate` : undefined}
              nextLabel={completed ? 'Get Your Certificate' : undefined}
            />
          </div>
        </FadeIn>
      </div>
    </>
  );
}