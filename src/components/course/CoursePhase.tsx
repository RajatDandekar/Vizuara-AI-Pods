'use client';

import { type ReactNode } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Badge from '@/components/ui/Badge';

type PhaseStatus = 'completed' | 'current' | 'locked';

interface CoursePhaseProps {
  phase: number;
  title: string;
  subtitle: string;
  status: PhaseStatus;
  statusLabel?: string;
  expanded: boolean;
  onToggle: () => void;
  children: ReactNode;
}

export default function CoursePhase({
  phase,
  title,
  subtitle,
  status,
  statusLabel,
  expanded,
  onToggle,
  children,
}: CoursePhaseProps) {
  const isClickable = status !== 'locked';

  return (
    <div
      className={`
        bg-card-bg border rounded-2xl overflow-hidden transition-all duration-300
        ${status === 'current' ? 'border-accent-blue/40 shadow-sm' : ''}
        ${status === 'completed' ? 'border-accent-green/30' : ''}
        ${status === 'locked' ? 'border-card-border opacity-75' : 'border-card-border'}
      `}
    >
      {/* Header — always visible, clickable to toggle */}
      <button
        onClick={isClickable ? onToggle : undefined}
        className={`
          w-full flex items-center gap-4 p-5 text-left transition-colors
          ${isClickable ? 'cursor-pointer hover:bg-gray-50/50' : 'cursor-default'}
        `}
      >
        {/* Phase indicator */}
        <div
          className={`
            w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 text-sm font-bold
            ${status === 'completed' ? 'bg-accent-green-light text-accent-green' : ''}
            ${status === 'current' ? 'bg-accent-blue-light text-accent-blue' : ''}
            ${status === 'locked' ? 'bg-gray-100 text-text-muted' : ''}
          `}
        >
          {status === 'completed' ? (
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
            </svg>
          ) : status === 'locked' ? (
            <svg className="w-4.5 h-4.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 10.5V6.75a4.5 4.5 0 10-9 0v3.75m-.75 11.25h10.5a2.25 2.25 0 002.25-2.25v-6.75a2.25 2.25 0 00-2.25-2.25H6.75a2.25 2.25 0 00-2.25 2.25v6.75a2.25 2.25 0 002.25 2.25z" />
            </svg>
          ) : (
            phase
          )}
        </div>

        {/* Title + subtitle */}
        <div className="flex-1 min-w-0">
          <h3 className={`font-semibold text-sm ${status === 'locked' ? 'text-text-muted' : 'text-foreground'}`}>
            {title}
          </h3>
          <p className="text-xs text-text-muted mt-0.5 truncate">{subtitle}</p>
        </div>

        {/* Status badge */}
        <div className="flex items-center gap-2 flex-shrink-0">
          {statusLabel && (
            <Badge
              variant={status === 'completed' ? 'green' : status === 'current' ? 'blue' : 'default'}
              size="sm"
            >
              {statusLabel}
            </Badge>
          )}

          {/* Chevron */}
          {isClickable && (
            <motion.svg
              animate={{ rotate: expanded ? 180 : 0 }}
              transition={{ duration: 0.2 }}
              className="w-4 h-4 text-text-muted"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
            </motion.svg>
          )}
        </div>
      </button>

      {/* Collapsible content */}
      <AnimatePresence initial={false}>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: [0.25, 0.1, 0.25, 1] }}
            className="overflow-hidden"
          >
            <div className="px-5 pb-5 pt-0">
              {children}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
