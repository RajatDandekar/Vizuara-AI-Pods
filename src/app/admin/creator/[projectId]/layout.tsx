'use client';

import { useEffect } from 'react';
import { useParams } from 'next/navigation';
import { useCreator } from '@/context/CreatorContext';
import CreatorSidebar from '@/components/creator/CreatorSidebar';

export default function ProjectLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const { projectId } = useParams<{ projectId: string }>();
  const { fetchProject, state } = useCreator();

  useEffect(() => {
    if (projectId) fetchProject(projectId);
  }, [projectId, fetchProject]);

  if (state.loading && !state.activeProject) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <p className="text-text-muted">Loading project...</p>
      </div>
    );
  }

  if (!state.activeProject) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <p className="text-text-muted">Project not found</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background flex">
      <CreatorSidebar project={state.activeProject} />
      <main className="flex-1 min-w-0 text-[15px] leading-relaxed">{children}</main>
    </div>
  );
}
