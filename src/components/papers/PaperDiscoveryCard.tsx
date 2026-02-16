'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import type { PaperState } from '@/types/paper';
import { getYearGradient, getArxivPdfUrl } from '@/lib/arxiv';
import { staggerItemVariants } from '@/components/animations/StaggerChildren';

interface PaperDiscoveryCardProps {
  paper: PaperState;
  concept: string;
  index: number;
}

export default function PaperDiscoveryCard({
  paper,
  concept,
  index,
}: PaperDiscoveryCardProps) {
  const gradient = getYearGradient(paper.year);
  const pdfUrl = getArxivPdfUrl(paper.arxivUrl);

  const summaryDone = paper.summary.status === 'complete';
  const archDone = paper.architecture.status === 'complete';
  const notebookDone = paper.notebook.status === 'complete';
  const anyStreaming =
    paper.summary.status === 'streaming' ||
    paper.architecture.status === 'streaming' ||
    paper.notebook.status === 'streaming';
  const sectionsComplete = [summaryDone, archDone, notebookDone].filter(Boolean).length;

  return (
    <motion.div variants={staggerItemVariants}>
      <Link
        href={`/discover/${paper.id}?concept=${encodeURIComponent(concept)}`}
        className="group block"
      >
        <div className="bg-card-bg border border-card-border rounded-2xl overflow-hidden shadow-sm hover:shadow-lg transition-all duration-300 hover:-translate-y-0.5">
          {/* Gradient header */}
          <div className={`relative bg-gradient-to-br ${gradient.from} ${gradient.to} px-5 py-5`}>
            {/* Paper number */}
            <div className="absolute top-3 right-3 w-8 h-8 rounded-full bg-white/20 backdrop-blur-sm flex items-center justify-center">
              <span className="text-sm font-bold text-white">{index + 1}</span>
            </div>

            <div className="flex items-center gap-2 mb-2">
              <span className="text-3xl font-bold text-white/90">{paper.year}</span>
              <span className="px-2 py-0.5 bg-white/20 backdrop-blur-sm rounded-md text-xs font-medium text-white">
                {paper.venue}
              </span>
            </div>

            {/* ArXiv thumbnail placeholder using first page preview */}
            {paper.arxivUrl && (
              <div className="mt-2 flex items-center gap-1.5">
                <svg className="w-3.5 h-3.5 text-white/70" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                </svg>
                <span className="text-xs text-white/60 truncate">arxiv.org</span>
              </div>
            )}
          </div>

          {/* Card body */}
          <div className="px-5 py-4">
            <h3 className="font-semibold text-foreground leading-snug line-clamp-2 mb-1.5 group-hover:text-accent-blue transition-colors">
              {paper.title}
            </h3>

            <p className="text-xs text-text-muted line-clamp-1 mb-2">
              {paper.authors.join(', ')}
            </p>

            <p className="text-sm text-text-secondary leading-relaxed line-clamp-2 mb-3">
              {paper.oneLiner}
            </p>

            {/* Footer: status + actions */}
            <div className="flex items-center justify-between">
              {/* Exploration status */}
              {sectionsComplete > 0 || anyStreaming ? (
                <div className="flex items-center gap-1.5">
                  {anyStreaming && (
                    <span className="inline-flex items-center gap-1 text-xs text-accent-blue font-medium">
                      <svg className="w-3 h-3 animate-spin" viewBox="0 0 24 24" fill="none">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                      </svg>
                      Generating...
                    </span>
                  )}
                  {!anyStreaming && sectionsComplete > 0 && (
                    <span className="text-xs text-accent-green font-medium">
                      {sectionsComplete}/3 complete
                    </span>
                  )}
                  {/* Progress dots */}
                  <div className="flex gap-1 ml-1">
                    <div className={`w-1.5 h-1.5 rounded-full ${summaryDone ? 'bg-accent-green' : 'bg-gray-200'}`} />
                    <div className={`w-1.5 h-1.5 rounded-full ${archDone ? 'bg-accent-green' : 'bg-gray-200'}`} />
                    <div className={`w-1.5 h-1.5 rounded-full ${notebookDone ? 'bg-accent-green' : 'bg-gray-200'}`} />
                  </div>
                </div>
              ) : (
                <span className="text-xs text-text-muted">Click to explore</span>
              )}

              {/* ArXiv PDF link — use button to avoid nested <a> inside <Link> */}
              {pdfUrl && (
                <button
                  type="button"
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    window.open(pdfUrl, '_blank', 'noopener,noreferrer');
                  }}
                  className="inline-flex items-center gap-1 text-xs text-text-muted hover:text-accent-blue transition-colors cursor-pointer"
                >
                  <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                  </svg>
                  PDF
                </button>
              )}
            </div>
          </div>
        </div>
      </Link>
    </motion.div>
  );
}
