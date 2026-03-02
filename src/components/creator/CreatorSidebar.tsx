'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import type { CreatorProject, ProjectStep, StepStatus } from '@/types/creator';

const STEPS: { key: ProjectStep; label: string; icon: string }[] = [
  { key: 'article', label: 'Article', icon: '1' },
  { key: 'figures', label: 'Figures', icon: '2' },
  { key: 'preview', label: 'Preview', icon: '3' },
  { key: 'export', label: 'Export', icon: '4' },
  { key: 'notebooks', label: 'Notebooks', icon: '5' },
  { key: 'case-study', label: 'Case Study', icon: '6' },
  { key: 'narration', label: 'Narration', icon: '7' },
  { key: 'publish', label: 'Publish', icon: '8' },
];

function statusStyles(status: StepStatus, isActive: boolean): string {
  if (isActive) {
    return 'bg-accent-blue text-white';
  }
  switch (status) {
    case 'complete':
      return 'bg-accent-green text-white';
    case 'in-progress':
      return 'bg-accent-blue-light text-accent-blue';
    case 'error':
      return 'bg-accent-red/10 text-accent-red';
    default:
      return 'bg-background text-text-muted';
  }
}

function checkIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 12 12" fill="none">
      <path
        d="M2 6L5 9L10 3"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

interface CreatorSidebarProps {
  project: CreatorProject;
}

export default function CreatorSidebar({ project }: CreatorSidebarProps) {
  const pathname = usePathname();

  return (
    <aside className="w-72 border-r border-card-border bg-card-bg min-h-screen flex flex-col">
      {/* Project header */}
      <div className="p-5 border-b border-card-border">
        <Link
          href="/admin/creator"
          className="text-base text-text-muted hover:text-accent-blue transition-colors"
        >
          &larr; All projects
        </Link>
        <h2 className="text-lg font-semibold text-foreground mt-2 line-clamp-2">
          {project.podTitle}
        </h2>
        <span className="text-base text-text-muted">{project.courseSlug}</span>
      </div>

      {/* Step navigation */}
      <nav className="flex-1 p-4 space-y-1">
        {STEPS.map(({ key, label, icon }) => {
          const stepState = project.steps.find((s) => s.step === key);
          const status = stepState?.status || 'pending';
          const isActive = pathname?.includes(`/${key}`);

          return (
            <Link
              key={key}
              href={`/admin/creator/${project.id}/${key}`}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors text-base
                ${isActive ? 'bg-accent-blue/5 text-foreground font-medium' : 'text-text-secondary hover:bg-background'}`}
            >
              <span
                className={`w-7 h-7 flex items-center justify-center rounded-full text-sm font-medium ${statusStyles(status, !!isActive)}`}
              >
                {status === 'complete' ? checkIcon() : icon}
              </span>
              {label}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-card-border">
        <p className="text-base text-text-muted">
          Created {new Date(project.createdAt).toLocaleDateString()}
        </p>
      </div>
    </aside>
  );
}
