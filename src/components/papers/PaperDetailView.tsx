'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Link from 'next/link';
import SectionStepper from './SectionStepper';
import PaperSummary from './PaperSummary';
import PaperArchitecture from './PaperArchitecture';
import PaperNotebook from './PaperNotebook';
import { getYearGradient, getArxivPdfUrl } from '@/lib/arxiv';
import type { PaperState } from '@/types/paper';

interface PaperDetailViewProps {
  paper: PaperState;
  concept: string;
}

type SectionTab = 'summary' | 'architecture' | 'notebook';

const SECTION_META: Record<SectionTab, { icon: React.ReactNode; title: string; description: string; color: string }> = {
  summary: {
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    ),
    title: 'Paper Summary',
    description: 'A concise breakdown of the paper\'s key contributions, methodology, and results.',
    color: 'blue',
  },
  architecture: {
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6A2.25 2.25 0 016 3.75h2.25A2.25 2.25 0 0110.5 6v2.25a2.25 2.25 0 01-2.25 2.25H6a2.25 2.25 0 01-2.25-2.25V6zM3.75 15.75A2.25 2.25 0 016 13.5h2.25a2.25 2.25 0 012.25 2.25V18a2.25 2.25 0 01-2.25 2.25H6A2.25 2.25 0 013.75 18v-2.25zM13.5 6a2.25 2.25 0 012.25-2.25H18A2.25 2.25 0 0120.25 6v2.25A2.25 2.25 0 0118 10.5h-2.25a2.25 2.25 0 01-2.25-2.25V6zM13.5 15.75a2.25 2.25 0 012.25-2.25H18a2.25 2.25 0 012.25 2.25V18A2.25 2.25 0 0118 20.25h-2.25A2.25 2.25 0 0113.5 18v-2.25z" />
      </svg>
    ),
    title: 'Architecture Deep Dive',
    description: 'Detailed walkthrough of the model architecture, components, and design decisions.',
    color: 'indigo',
  },
  notebook: {
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
      </svg>
    ),
    title: 'Colab Notebook',
    description: 'An executable Google Colab notebook with code, explanations, and exercises.',
    color: 'emerald',
  },
};

export default function PaperDetailView({ paper, concept }: PaperDetailViewProps) {
  const [activeTab, setActiveTab] = useState<SectionTab>('summary');
  // Track which sections the user has explicitly triggered
  const [triggered, setTriggered] = useState<Set<SectionTab>>(new Set(['summary']));

  const gradient = getYearGradient(paper.year);
  const pdfUrl = getArxivPdfUrl(paper.arxivUrl);

  function handleGenerate(section: SectionTab) {
    setTriggered((prev) => new Set([...prev, section]));
    setActiveTab(section);
  }

  const steps = (['summary', 'architecture', 'notebook'] as SectionTab[]).map((key) => ({
    key,
    label: SECTION_META[key].title,
    status: paper[key].status,
    generated: triggered.has(key),
  }));

  return (
    <div>
      {/* Back link */}
      <Link
        href={`/discover?concept=${encodeURIComponent(concept)}`}
        className="inline-flex items-center gap-1.5 text-sm text-text-muted hover:text-foreground transition-colors mb-6"
      >
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
        </svg>
        Back to papers
      </Link>

      {/* Paper header */}
      <div className={`bg-gradient-to-br ${gradient.from} ${gradient.to} rounded-2xl px-6 py-6 sm:px-8 sm:py-8 mb-8`}>
        <div className="flex items-center gap-3 mb-3">
          <span className="text-3xl font-bold text-white/90">{paper.year}</span>
          <span className="px-2.5 py-1 bg-white/20 backdrop-blur-sm rounded-lg text-xs font-medium text-white">
            {paper.venue}
          </span>
        </div>

        <h1 className="text-xl sm:text-2xl font-bold text-white leading-snug mb-3">
          {paper.title}
        </h1>

        <p className="text-sm text-white/70 mb-4">
          {paper.authors.join(', ')}
        </p>

        <p className="text-sm text-white/80 leading-relaxed mb-4">
          <span className="font-semibold text-white">Why this matters:</span>{' '}
          {paper.significance}
        </p>

        {pdfUrl && (
          <a
            href={pdfUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-white/20 backdrop-blur-sm rounded-lg text-xs font-medium text-white hover:bg-white/30 transition-colors"
          >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
            </svg>
            Read Original Paper (PDF)
          </a>
        )}
      </div>

      {/* Section stepper — clickable tabs */}
      <SectionStepper steps={steps} activeKey={activeTab} onStepClick={(key) => setActiveTab(key as SectionTab)} />

      {/* Tab content — only the active tab is visible, but triggered sections stay mounted */}
      <div className="relative">
        {/* Summary */}
        <div className={activeTab === 'summary' ? '' : 'hidden'}>
          <SectionCard section="summary">
            <PaperSummary paper={paper} concept={concept} />
          </SectionCard>
        </div>

        {/* Architecture */}
        <div className={activeTab === 'architecture' ? '' : 'hidden'}>
          {triggered.has('architecture') ? (
            <SectionCard section="architecture">
              <PaperArchitecture paper={paper} concept={concept} />
            </SectionCard>
          ) : (
            <GenerateCTA section="architecture" onGenerate={() => handleGenerate('architecture')} />
          )}
        </div>

        {/* Notebook */}
        <div className={activeTab === 'notebook' ? '' : 'hidden'}>
          {triggered.has('notebook') ? (
            <SectionCard section="notebook">
              <PaperNotebook paper={paper} concept={concept} />
            </SectionCard>
          ) : (
            <GenerateCTA section="notebook" onGenerate={() => handleGenerate('notebook')} />
          )}
        </div>
      </div>
    </div>
  );
}

/** Wraps section content in a reading-friendly card */
function SectionCard({ section, children }: { section: SectionTab; children: React.ReactNode }) {
  const meta = SECTION_META[section];
  const borderColor = section === 'summary' ? 'border-l-blue-500' : section === 'architecture' ? 'border-l-indigo-500' : 'border-l-emerald-500';

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={section}
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        <div className={`bg-card-bg border border-card-border border-l-4 ${borderColor} rounded-2xl shadow-sm overflow-hidden`}>
          {/* Section header */}
          <div className="px-6 py-4 border-b border-card-border bg-gray-50/50">
            <div className="flex items-center gap-2.5">
              <span className="text-text-muted">{meta.icon}</span>
              <h2 className="text-base font-semibold text-foreground">{meta.title}</h2>
            </div>
          </div>

          {/* Content area — comfortable reading width and spacing */}
          <div className="px-6 py-5 sm:px-8 sm:py-6">
            {children}
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}

/** CTA card shown for sections that haven't been generated yet */
function GenerateCTA({ section, onGenerate }: { section: SectionTab; onGenerate: () => void }) {
  const meta = SECTION_META[section];
  const btnColor = section === 'architecture'
    ? 'bg-indigo-600 hover:bg-indigo-700 shadow-indigo-200'
    : 'bg-emerald-600 hover:bg-emerald-700 shadow-emerald-200';

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className="bg-card-bg border border-card-border rounded-2xl p-8 sm:p-10 text-center">
        <div className="w-14 h-14 rounded-2xl bg-gray-100 flex items-center justify-center mx-auto mb-4 text-text-muted">
          {meta.icon}
        </div>
        <h3 className="text-lg font-semibold text-foreground mb-2">{meta.title}</h3>
        <p className="text-sm text-text-secondary max-w-md mx-auto mb-6 leading-relaxed">
          {meta.description}
        </p>
        <button
          type="button"
          onClick={onGenerate}
          className={`inline-flex items-center gap-2 px-5 py-2.5 ${btnColor} text-white text-sm font-medium rounded-xl shadow-lg transition-all cursor-pointer hover:-translate-y-0.5`}
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.455 2.456L21.75 6l-1.036.259a3.375 3.375 0 00-2.455 2.456z" />
          </svg>
          Generate {meta.title}
        </button>
      </div>
    </motion.div>
  );
}
