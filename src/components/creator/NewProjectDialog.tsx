'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';

interface Course {
  slug: string;
  title: string;
}

interface NewProjectDialogProps {
  onClose: () => void;
}

export default function NewProjectDialog({ onClose }: NewProjectDialogProps) {
  const router = useRouter();
  const [concept, setConcept] = useState('');
  const [podTitle, setPodTitle] = useState('');
  const [podSlug, setPodSlug] = useState('');
  const [courseSlug, setCourseSlug] = useState('');
  const [courses, setCourses] = useState<Course[]>([]);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    // Load courses from catalog
    fetch('/api/admin/creator/projects')
      .then(() => {
        // Also fetch catalog for course list
        return fetch('/content/courses/catalog.json');
      })
      .then((res) => res.json())
      .then((data) => {
        if (data.courses) {
          setCourses(
            data.courses.map((c: { slug: string; title: string }) => ({
              slug: c.slug,
              title: c.title,
            }))
          );
          if (data.courses.length > 0) {
            setCourseSlug(data.courses[0].slug);
          }
        }
      })
      .catch(() => {
        // Fallback: hardcode known courses
        setCourses([
          { slug: 'diffusion-models', title: 'Principles of Diffusion Models' },
          { slug: 'rl-from-scratch', title: 'RL From Scratch' },
          { slug: 'build-llm', title: 'Build an LLM' },
          { slug: 'vlms-from-scratch', title: 'VLMs From Scratch' },
        ]);
        setCourseSlug('diffusion-models');
      });
  }, []);

  // Auto-generate slug from title
  useEffect(() => {
    if (podTitle) {
      setPodSlug(
        podTitle
          .toLowerCase()
          .replace(/[^a-z0-9\s-]/g, '')
          .replace(/\s+/g, '-')
          .replace(/-+/g, '-')
          .slice(0, 60)
      );
    }
  }, [podTitle]);

  const handleCreate = async () => {
    if (!concept || !podTitle || !podSlug || !courseSlug) {
      setError('All fields are required');
      return;
    }

    setCreating(true);
    setError('');

    try {
      const res = await fetch('/api/admin/creator/projects', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ concept, podTitle, podSlug, courseSlug }),
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || 'Failed to create project');
      }

      const { project } = await res.json();
      router.push(`/admin/creator/${project.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create project');
      setCreating(false);
    }
  };

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center">
        {/* Backdrop */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="absolute inset-0 bg-black/40"
          onClick={onClose}
        />

        {/* Dialog */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 10 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 10 }}
          className="relative bg-card-bg border border-card-border rounded-2xl shadow-xl
                     w-full max-w-lg p-6 mx-4"
        >
          <h2 className="text-2xl font-bold text-foreground mb-1">New Pod</h2>
          <p className="text-base text-text-muted mb-6">
            Enter the concept you want to teach
          </p>

          <div className="space-y-4">
            {/* Concept */}
            <div>
              <label className="block text-base font-medium text-foreground mb-1.5">
                Concept / Topic
              </label>
              <textarea
                value={concept}
                onChange={(e) => setConcept(e.target.value)}
                placeholder="e.g., Understanding Variational Autoencoders from first principles"
                rows={3}
                className="w-full px-3 py-2.5 border border-card-border rounded-lg bg-background
                           text-foreground placeholder-text-muted resize-none
                           focus:outline-none focus:ring-2 focus:ring-accent-blue/30 focus:border-accent-blue"
              />
            </div>

            {/* Pod Title */}
            <div>
              <label className="block text-base font-medium text-foreground mb-1.5">
                Pod Title
              </label>
              <input
                type="text"
                value={podTitle}
                onChange={(e) => setPodTitle(e.target.value)}
                placeholder="e.g., Variational Autoencoders from Scratch"
                className="w-full px-3 py-2.5 border border-card-border rounded-lg bg-background
                           text-foreground placeholder-text-muted
                           focus:outline-none focus:ring-2 focus:ring-accent-blue/30 focus:border-accent-blue"
              />
            </div>

            {/* Pod Slug (auto-generated) */}
            <div>
              <label className="block text-base font-medium text-foreground mb-1.5">
                Pod Slug
              </label>
              <input
                type="text"
                value={podSlug}
                onChange={(e) => setPodSlug(e.target.value)}
                placeholder="vae-from-scratch"
                className="w-full px-3 py-2.5 border border-card-border rounded-lg bg-background
                           text-foreground placeholder-text-muted font-mono text-sm
                           focus:outline-none focus:ring-2 focus:ring-accent-blue/30 focus:border-accent-blue"
              />
            </div>

            {/* Course Selection */}
            <div>
              <label className="block text-base font-medium text-foreground mb-1.5">
                Parent Course
              </label>
              <select
                value={courseSlug}
                onChange={(e) => setCourseSlug(e.target.value)}
                className="w-full px-3 py-2.5 border border-card-border rounded-lg bg-background
                           text-foreground
                           focus:outline-none focus:ring-2 focus:ring-accent-blue/30 focus:border-accent-blue"
              >
                {courses.map((c) => (
                  <option key={c.slug} value={c.slug}>
                    {c.title}
                  </option>
                ))}
              </select>
            </div>

            {error && (
              <p className="text-base text-accent-red">{error}</p>
            )}
          </div>

          <div className="flex justify-end gap-3 mt-6">
            <button
              onClick={onClose}
              className="px-4 py-2 text-text-secondary hover:text-foreground transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleCreate}
              disabled={creating}
              className="px-5 py-2 bg-accent-blue text-white rounded-lg font-medium
                         hover:bg-accent-blue/90 transition-colors disabled:opacity-50"
            >
              {creating ? 'Creating...' : 'Create Pod'}
            </button>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
