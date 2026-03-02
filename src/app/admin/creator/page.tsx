'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useCreator } from '@/context/CreatorContext';
import FadeIn from '@/components/animations/FadeIn';
import NewProjectDialog from '@/components/creator/NewProjectDialog';
import type { CreatorProject, StepStatus } from '@/types/creator';

function getOverallStatus(project: CreatorProject): { label: string; color: string } {
  const completedCount = project.steps.filter((s) => s.status === 'complete').length;
  if (project.publishedAt) return { label: 'Published', color: 'text-accent-green' };
  if (completedCount === 0) return { label: 'Not started', color: 'text-text-muted' };
  return { label: `${completedCount}/8 steps`, color: 'text-accent-blue' };
}

function statusDot(status: StepStatus) {
  switch (status) {
    case 'complete':
      return 'bg-accent-green';
    case 'in-progress':
      return 'bg-accent-blue';
    case 'error':
      return 'bg-accent-red';
    default:
      return 'bg-card-border';
  }
}

export default function CreatorDashboard() {
  const router = useRouter();
  const { state, fetchProjects } = useCreator();
  const [showNewProject, setShowNewProject] = useState(false);

  useEffect(() => {
    fetchProjects();
  }, [fetchProjects]);

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-6xl mx-auto px-6 py-10">
        <FadeIn>
          <div className="flex items-center justify-between mb-8">
            <div>
              <h1 className="text-3xl font-bold text-foreground">Pod Creator</h1>
              <p className="text-text-secondary text-lg mt-1">
                Create complete pods with articles, notebooks, and more
              </p>
            </div>
            <button
              onClick={() => setShowNewProject(true)}
              className="px-5 py-2.5 bg-accent-blue text-white rounded-lg font-medium
                         hover:bg-accent-blue/90 transition-colors"
            >
              New Pod
            </button>
          </div>
        </FadeIn>

        {state.loading && !state.projects.length ? (
          <div className="text-center py-20 text-text-muted">Loading projects...</div>
        ) : state.projects.length === 0 ? (
          <FadeIn delay={0.1}>
            <div className="text-center py-20 border border-dashed border-card-border rounded-xl">
              <p className="text-text-muted text-lg mb-4">No projects yet</p>
              <button
                onClick={() => setShowNewProject(true)}
                className="px-5 py-2.5 bg-accent-blue text-white rounded-lg font-medium
                           hover:bg-accent-blue/90 transition-colors"
              >
                Create your first pod
              </button>
            </div>
          </FadeIn>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {state.projects.map((project, i) => {
              const overall = getOverallStatus(project);
              return (
                <FadeIn key={project.id} delay={i * 0.05}>
                  <button
                    onClick={() =>
                      router.push(`/admin/creator/${project.id}`)
                    }
                    className="w-full text-left p-5 bg-card-bg border border-card-border rounded-xl
                               hover:border-accent-blue/50 hover:shadow-md transition-all"
                  >
                    <div className="flex items-start justify-between mb-3">
                      <h3 className="font-semibold text-foreground line-clamp-1">
                        {project.podTitle}
                      </h3>
                      <span className={`text-sm font-medium ${overall.color}`}>
                        {overall.label}
                      </span>
                    </div>
                    <p className="text-base text-text-muted mb-3 line-clamp-2">
                      {project.concept}
                    </p>
                    <div className="flex items-center gap-3 mb-3">
                      <span className="text-sm text-text-muted bg-background px-2 py-0.5 rounded">
                        {project.courseSlug}
                      </span>
                      <span className="text-sm text-text-muted">
                        {new Date(project.createdAt).toLocaleDateString()}
                      </span>
                    </div>
                    <div className="flex gap-1">
                      {project.steps.map((step) => (
                        <div
                          key={step.step}
                          className={`h-1.5 flex-1 rounded-full ${statusDot(step.status)}`}
                          title={`${step.step}: ${step.status}`}
                        />
                      ))}
                    </div>
                  </button>
                </FadeIn>
              );
            })}
          </div>
        )}

        {showNewProject && (
          <NewProjectDialog onClose={() => setShowNewProject(false)} />
        )}
      </div>
    </div>
  );
}
