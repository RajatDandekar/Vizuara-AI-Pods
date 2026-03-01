'use client';

import { useState } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { motion } from 'framer-motion';
import type { FreePodShowcase } from '@/types/course';

const difficultyColor: Record<string, string> = {
  beginner: 'bg-emerald-50 text-emerald-700 border-emerald-200',
  intermediate: 'bg-blue-50 text-blue-700 border-blue-200',
  advanced: 'bg-amber-50 text-amber-700 border-amber-200',
};

const tabs = ['article', 'notebooks', 'case-study'] as const;
type Tab = (typeof tabs)[number];

const tabLabels: Record<Tab, string> = {
  article: 'Article',
  notebooks: 'Notebooks',
  'case-study': 'Case Study',
};

interface FreeModuleCardProps {
  pod: FreePodShowcase;
  accentColor: string; // tailwind color like 'blue' | 'emerald' | 'violet'
}

export default function FreeModuleCard({ pod, accentColor }: FreeModuleCardProps) {
  const [activeTab, setActiveTab] = useState<Tab>('article');

  const accentClasses: Record<string, { badge: string; tab: string; cta: string; ring: string }> = {
    blue: {
      badge: 'from-blue-500 to-indigo-500',
      tab: 'bg-blue-600 text-white',
      cta: 'bg-blue-600 hover:bg-blue-700 text-white',
      ring: 'hover:ring-blue-200',
    },
    emerald: {
      badge: 'from-emerald-500 to-teal-500',
      tab: 'bg-emerald-600 text-white',
      cta: 'bg-emerald-600 hover:bg-emerald-700 text-white',
      ring: 'hover:ring-emerald-200',
    },
    violet: {
      badge: 'from-violet-500 to-purple-500',
      tab: 'bg-violet-600 text-white',
      cta: 'bg-violet-600 hover:bg-violet-700 text-white',
      ring: 'hover:ring-violet-200',
    },
    amber: {
      badge: 'from-amber-500 to-orange-500',
      tab: 'bg-amber-600 text-white',
      cta: 'bg-amber-600 hover:bg-amber-700 text-white',
      ring: 'hover:ring-amber-200',
    },
  };

  const accent = accentClasses[accentColor] || accentClasses.blue;

  return (
    <motion.div
      className={`bg-white rounded-2xl overflow-hidden flex flex-col border border-slate-200 shadow-sm hover:shadow-xl hover:ring-2 ${accent.ring} transition-all duration-300`}
      whileHover={{ y: -4 }}
      transition={{ type: 'spring', stiffness: 300, damping: 25 }}
    >
      {/* Hero image */}
      <div className="relative w-full aspect-[16/9] overflow-hidden bg-slate-50">
        {pod.thumbnail ? (
          <Image
            src={pod.thumbnail}
            alt={pod.title}
            fill
            className="object-cover"
            sizes="(max-width: 640px) 90vw, 380px"
          />
        ) : (
          <div className="w-full h-full bg-gradient-to-br from-slate-100 to-slate-50" />
        )}
        <div className="absolute inset-0 bg-gradient-to-t from-white/80 via-transparent to-transparent" />

        {/* FREE badge */}
        <div className="absolute top-3 left-3">
          <span className={`bg-gradient-to-r ${accent.badge} text-white text-[11px] font-bold px-2.5 py-1 rounded-full shadow-sm`}>
            FREE
          </span>
        </div>
      </div>

      {/* Header */}
      <div className="px-5 pt-4 pb-2">
        <p className="text-[11px] text-slate-400 font-medium mb-1">{pod.courseTitle}</p>
        <h3 className="font-bold text-slate-900 text-lg leading-snug mb-2">{pod.title}</h3>
        <div className="flex items-center gap-2 flex-wrap">
          <span className={`text-[11px] font-medium px-2 py-0.5 rounded-full border ${difficultyColor[pod.difficulty]}`}>
            {pod.difficulty}
          </span>
          <span className="text-[11px] text-slate-400">~{pod.estimatedHours}h</span>
          <span className="text-[11px] text-slate-400">{pod.notebooks.length} notebooks</span>
          {pod.caseStudy && (
            <span className="text-[11px] text-slate-400">+ case study</span>
          )}
        </div>
      </div>

      {/* Tab bar */}
      <div className="px-5 pt-3">
        <div className="flex gap-1 bg-slate-100 rounded-lg p-0.5">
          {tabs.map((tab) => {
            if (tab === 'case-study' && !pod.caseStudy) return null;
            return (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`flex-1 text-[12px] font-medium py-1.5 rounded-md transition-all ${
                  activeTab === tab ? accent.tab : 'text-slate-500 hover:text-slate-700'
                }`}
              >
                {tabLabels[tab]}
              </button>
            );
          })}
        </div>
      </div>

      {/* Tab content */}
      <div className="px-5 pt-3 pb-4 flex-1 flex flex-col min-h-[180px]">
        {activeTab === 'article' && (
          <div className="flex flex-col flex-1">
            <p className="text-sm text-slate-600 leading-relaxed mb-4 line-clamp-3">
              {pod.description}
            </p>
            <div className="mt-auto">
              <Link
                href={`/courses/${pod.courseSlug}/${pod.podSlug}/article`}
                className="text-sm font-medium text-blue-600 hover:text-blue-700 inline-flex items-center gap-1 transition-colors"
              >
                Read Full Article
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
                </svg>
              </Link>
            </div>
          </div>
        )}

        {activeTab === 'notebooks' && (
          <div className="flex flex-col flex-1">
            <ul className="space-y-1.5 mb-4">
              {pod.notebooks.map((nb) => (
                <li key={nb.slug} className="flex items-start gap-2">
                  <svg className="w-3.5 h-3.5 text-emerald-500 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M17.25 6.75L22.5 12l-5.25 5.25m-10.5 0L1.5 12l5.25-5.25m7.5-3l-4.5 16.5" />
                  </svg>
                  <a
                    href={nb.colabUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm text-slate-700 hover:text-blue-600 transition-colors leading-snug"
                  >
                    {nb.title}
                  </a>
                </li>
              ))}
            </ul>
            <div className="mt-auto">
              <Link
                href={`/courses/${pod.courseSlug}/${pod.podSlug}/practice`}
                className="text-sm font-medium text-emerald-600 hover:text-emerald-700 inline-flex items-center gap-1 transition-colors"
              >
                View All Notebooks
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
                </svg>
              </Link>
            </div>
          </div>
        )}

        {activeTab === 'case-study' && pod.caseStudy && (
          <div className="flex flex-col flex-1">
            <p className="text-[11px] font-semibold text-amber-600 uppercase tracking-wider mb-1">
              {pod.caseStudy.company} &middot; {pod.caseStudy.industry}
            </p>
            <p className="text-sm text-slate-600 leading-relaxed mb-4 line-clamp-3">
              {pod.caseStudy.description}
            </p>
            <div className="mt-auto">
              <Link
                href={`/courses/${pod.courseSlug}/${pod.podSlug}/case-study`}
                className="text-sm font-medium text-amber-600 hover:text-amber-700 inline-flex items-center gap-1 transition-colors"
              >
                Start Case Study
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
                </svg>
              </Link>
            </div>
          </div>
        )}
      </div>

      {/* Primary CTA */}
      <div className="px-5 pb-5">
        <Link
          href={`/courses/${pod.courseSlug}/${pod.podSlug}`}
          className={`block w-full text-center text-sm font-semibold py-2.5 rounded-xl ${accent.cta} transition-colors`}
        >
          Explore This Module Free
        </Link>
      </div>
    </motion.div>
  );
}
