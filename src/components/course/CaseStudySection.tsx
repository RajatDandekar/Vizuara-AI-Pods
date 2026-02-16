'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import Button from '@/components/ui/Button';
import Badge from '@/components/ui/Badge';
import type { CaseStudyMeta } from '@/types/course';

interface CaseStudySectionProps {
  slug: string;
  caseStudy: CaseStudyMeta;
  unlocked: boolean;
}

export default function CaseStudySection({ slug, caseStudy, unlocked }: CaseStudySectionProps) {
  return (
    <div className="mt-10">
      {/* Section header */}
      <div className="flex items-center gap-3 mb-4">
        <h2 className="text-sm font-semibold text-text-muted uppercase tracking-wider">
          Apply Your Knowledge
        </h2>
        <div className="flex-1 h-px bg-card-border" />
        <Badge variant="amber" size="sm">Industry Project</Badge>
      </div>

      {/* Main card */}
      <div className={`relative rounded-2xl overflow-hidden ${!unlocked ? 'opacity-60' : ''}`}>
        {/* Gradient border */}
        <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-amber-400/20 via-orange-400/10 to-amber-500/20 pointer-events-none" />

        <div className="relative bg-card-bg border border-amber-200/60 rounded-2xl p-6 sm:p-8">
          {/* Top section: Icon + context */}
          <div className="flex items-start gap-4 mb-6">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-amber-50 to-orange-50 border border-amber-200/60 flex items-center justify-center flex-shrink-0">
              {unlocked ? (
                <svg className="w-6 h-6 text-accent-amber" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M20.25 14.15v4.25c0 1.094-.787 2.036-1.872 2.18-2.087.277-4.216.42-6.378.42s-4.291-.143-6.378-.42c-1.085-.144-1.872-1.086-1.872-2.18v-4.25m16.5 0a2.18 2.18 0 00.75-1.661V8.706c0-1.081-.768-2.015-1.837-2.175a48.114 48.114 0 00-3.413-.387m4.5 8.006c-.194.165-.42.295-.673.38A23.978 23.978 0 0112 15.75c-2.648 0-5.195-.429-7.577-1.22a2.016 2.016 0 01-.673-.38m0 0A2.18 2.18 0 013 12.489V8.706c0-1.081.768-2.015 1.837-2.175a48.111 48.111 0 013.413-.387m7.5 0V5.25A2.25 2.25 0 0013.5 3h-3a2.25 2.25 0 00-2.25 2.25v.894m7.5 0a48.667 48.667 0 00-7.5 0M12 12.75h.008v.008H12v-.008z" />
                </svg>
              ) : (
                <svg className="w-6 h-6 text-text-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 10.5V6.75a4.5 4.5 0 10-9 0v3.75m-.75 11.25h10.5a2.25 2.25 0 002.25-2.25v-6.75a2.25 2.25 0 00-2.25-2.25H6.75a2.25 2.25 0 00-2.25 2.25v6.75a2.25 2.25 0 002.25 2.25z" />
                </svg>
              )}
            </div>
            <div className="flex-1 min-w-0">
              <h3 className="text-lg font-bold text-foreground mb-1">
                Real-World Case Study
              </h3>
              {unlocked ? (
                <p className="text-sm text-text-secondary leading-relaxed">
                  Take what you learned and apply it to an industry-grade problem.
                  This case study walks you step by step through business context,
                  technical formulation, a hands-on implementation notebook, and
                  production system design.
                </p>
              ) : (
                <p className="text-sm text-text-muted leading-relaxed">
                  Complete all practice notebooks to unlock this industry case study.
                  You will get a real-world project brief, a Colab implementation
                  notebook, and a production system design walkthrough.
                </p>
              )}
            </div>
          </div>

          {/* Case study card — clickable only when unlocked */}
          {unlocked ? (
            <Link href={`/courses/${slug}/case-study`}>
              <motion.div
                whileHover={{ y: -2 }}
                transition={{ duration: 0.2 }}
                className="bg-gradient-to-br from-gray-50 to-amber-50/30 border border-card-border rounded-xl p-5 sm:p-6 cursor-pointer hover:shadow-md hover:border-amber-300/50 transition-all"
              >
                <CaseStudyCardContent caseStudy={caseStudy} />
                {/* CTA row */}
                <div className="flex items-center justify-between mt-5">
                  <span className="text-sm font-medium text-accent-amber flex items-center gap-1.5">
                    View case study
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
                    </svg>
                  </span>
                </div>
              </motion.div>
            </Link>
          ) : (
            <div className="bg-gray-50 border border-card-border rounded-xl p-5 sm:p-6">
              <CaseStudyCardContent caseStudy={caseStudy} />
            </div>
          )}

          {/* Quick-access buttons — only when unlocked */}
          {unlocked && (
            <div className="flex items-center gap-3 mt-4 flex-wrap">
              {caseStudy.notebookPath && (
                <a href={caseStudy.notebookPath} download>
                  <Button variant="secondary" size="sm">
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
                    </svg>
                    Download Notebook
                  </Button>
                </a>
              )}
              {caseStudy.colabUrl && (
                <a href={caseStudy.colabUrl} target="_blank" rel="noopener noreferrer">
                  <Button variant="secondary" size="sm">
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 6H5.25A2.25 2.25 0 003 8.25v10.5A2.25 2.25 0 005.25 21h10.5A2.25 2.25 0 0018 18.75V10.5m-10.5 6L21 3m0 0h-5.25M21 3v5.25" />
                    </svg>
                    Open in Colab
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
          )}
        </div>
      </div>
    </div>
  );
}

/** Shared content for the inner case study preview card */
function CaseStudyCardContent({ caseStudy }: { caseStudy: CaseStudyMeta }) {
  return (
    <>
      {/* Title row */}
      <div className="mb-3">
        <div className="flex items-center gap-2 mb-2 flex-wrap">
          {caseStudy.company && (
            <span className="text-xs font-medium text-accent-amber bg-accent-amber-light px-2 py-0.5 rounded-full">
              {caseStudy.company}
            </span>
          )}
          {caseStudy.industry && (
            <span className="text-xs text-text-muted">
              {caseStudy.industry}
            </span>
          )}
        </div>
        <h4 className="text-base sm:text-lg font-semibold text-foreground leading-snug">
          {caseStudy.title}
        </h4>
        {caseStudy.subtitle && (
          <p className="text-sm text-text-secondary mt-1">
            {caseStudy.subtitle}
          </p>
        )}
      </div>

      {/* Description */}
      {caseStudy.description && (
        <p className="text-sm text-text-secondary leading-relaxed mb-4 line-clamp-3">
          {caseStudy.description}
        </p>
      )}

      {/* What you'll build */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <div className="flex items-center gap-2.5 text-sm text-text-secondary">
          <div className="w-7 h-7 rounded-lg bg-blue-50 flex items-center justify-center flex-shrink-0">
            <svg className="w-3.5 h-3.5 text-accent-blue" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <span>Problem Brief</span>
        </div>
        <div className="flex items-center gap-2.5 text-sm text-text-secondary">
          <div className="w-7 h-7 rounded-lg bg-green-50 flex items-center justify-center flex-shrink-0">
            <svg className="w-3.5 h-3.5 text-accent-green" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M17.25 6.75L22.5 12l-5.25 5.25m-10.5 0L1.5 12l5.25-5.25m7.5-3l-4.5 16.5" />
            </svg>
          </div>
          <span>Colab Notebook</span>
        </div>
        <div className="flex items-center gap-2.5 text-sm text-text-secondary">
          <div className="w-7 h-7 rounded-lg bg-purple-50 flex items-center justify-center flex-shrink-0">
            <svg className="w-3.5 h-3.5 text-purple-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 14.25h13.5m-13.5 0a3 3 0 01-3-3m3 3a3 3 0 100 6h13.5a3 3 0 100-6m-16.5-3a3 3 0 013-3h13.5a3 3 0 013 3m-19.5 0a4.5 4.5 0 01.9-2.7L5.737 5.1a3.375 3.375 0 012.7-1.35h7.126c1.062 0 2.062.5 2.7 1.35l2.587 3.45a4.5 4.5 0 01.9 2.7" />
            </svg>
          </div>
          <span>Production Design</span>
        </div>
      </div>
    </>
  );
}
