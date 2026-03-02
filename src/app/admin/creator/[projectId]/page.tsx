'use client';

import { useEffect } from 'react';
import { useRouter, useParams } from 'next/navigation';
import { useCreator } from '@/context/CreatorContext';

export default function ProjectPage() {
  const router = useRouter();
  const { projectId } = useParams<{ projectId: string }>();
  const { state } = useCreator();
  const project = state.activeProject;

  useEffect(() => {
    if (project) {
      // Redirect to current step
      router.replace(`/admin/creator/${projectId}/${project.currentStep}`);
    }
  }, [project, projectId, router]);

  return (
    <div className="flex items-center justify-center h-full">
      <p className="text-text-muted">Redirecting...</p>
    </div>
  );
}
