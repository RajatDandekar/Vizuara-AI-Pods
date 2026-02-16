'use client';

import { useEffect, useRef, useCallback, useState } from 'react';
import { motion } from 'framer-motion';
import Button from '@/components/ui/Button';
import { useStreamingResponse } from '@/hooks/useStreamingResponse';
import type { PaperState, NotebookDocument } from '@/types/paper';

interface PaperNotebookProps {
  paper: PaperState;
  concept: string;
}

export default function PaperNotebook({
  paper,
  concept,
}: PaperNotebookProps) {
  const { startStreaming } = useStreamingResponse({
    endpoint: '/api/notebook',
    paperId: paper.id,
    section: 'notebook',
  });

  const hasStartedRef = useRef(false);
  const [uploading, setUploading] = useState(false);

  useEffect(() => {
    if (paper.notebook.status === 'idle' && !hasStartedRef.current) {
      hasStartedRef.current = true;
      startStreaming({
        concept,
        paper: {
          title: paper.title,
          authors: paper.authors,
          year: paper.year,
          venue: paper.venue,
        },
      });
    }
  }, [paper.notebook.status, concept, paper, startStreaming]);

  const handleDownload = useCallback((notebookJson: NotebookDocument) => {
    const blob = new Blob([JSON.stringify(notebookJson, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${paper.id}.ipynb`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [paper.id]);

  const handleOpenInColab = useCallback(async (notebookJson: NotebookDocument) => {
    setUploading(true);
    try {
      const res = await fetch('/api/upload-notebook', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          notebook: notebookJson,
          filename: `${paper.id}.ipynb`,
        }),
      });
      const data = await res.json();
      if (data.colabUrl) {
        window.open(data.colabUrl, '_blank');
      } else {
        handleDownload(notebookJson);
      }
    } catch {
      handleDownload(notebookJson);
    } finally {
      setUploading(false);
    }
  }, [paper.id, handleDownload]);

  // Generating state — progress card
  if (paper.notebook.status === 'streaming' || paper.notebook.status === 'loading') {
    return (
      <div className="text-center py-8">
        <div className="w-16 h-16 rounded-2xl bg-emerald-50 flex items-center justify-center mx-auto mb-5">
          <svg className="w-7 h-7 text-emerald-600 animate-spin" viewBox="0 0 24 24" fill="none">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
        </div>
        <h3 className="text-base font-semibold text-foreground mb-2">Building your notebook...</h3>
        <p className="text-sm text-text-secondary max-w-sm mx-auto leading-relaxed mb-4">
          Creating code cells, markdown explanations, and exercises. This may take a minute.
        </p>
        {/* Simple progress indicator based on content length */}
        <div className="max-w-xs mx-auto">
          <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-emerald-500 rounded-full"
              initial={{ width: '5%' }}
              animate={{ width: paper.notebook.content.length > 2000 ? '80%' : paper.notebook.content.length > 500 ? '45%' : '15%' }}
              transition={{ duration: 1.5, ease: 'easeOut' }}
            />
          </div>
          <p className="text-xs text-text-muted mt-2">Generating cells...</p>
        </div>
      </div>
    );
  }

  // Complete state — download / Colab buttons
  if (paper.notebook.status === 'complete' && paper.notebook.notebookJson) {
    const cellCount = paper.notebook.notebookJson.cells.length;
    const codeCells = paper.notebook.notebookJson.cells.filter(c => c.cell_type === 'code').length;

    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.97 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.3 }}
        className="text-center py-6"
      >
        <div className="w-16 h-16 rounded-2xl bg-emerald-50 flex items-center justify-center mx-auto mb-5">
          <svg className="w-7 h-7 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </div>

        <h3 className="text-base font-semibold text-foreground mb-1">Notebook Ready</h3>
        <p className="text-sm text-text-secondary mb-6">
          {cellCount} cells ({codeCells} code, {cellCount - codeCells} markdown) with explanations and exercises.
        </p>

        <div className="flex items-center justify-center gap-3">
          <Button
            onClick={() => handleOpenInColab(paper.notebook.notebookJson!)}
            size="md"
            disabled={uploading}
          >
            {uploading ? (
              <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
            ) : (
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
              </svg>
            )}
            {uploading ? 'Uploading to Drive...' : 'Open in Colab'}
          </Button>
          <Button
            variant="secondary"
            size="md"
            onClick={() => handleDownload(paper.notebook.notebookJson!)}
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            Download .ipynb
          </Button>
        </div>
      </motion.div>
    );
  }

  // Error state
  if (paper.notebook.status === 'error') {
    return (
      <div className="text-center py-6">
        <div className="w-14 h-14 rounded-2xl bg-red-50 flex items-center justify-center mx-auto mb-4">
          <svg className="w-6 h-6 text-accent-red" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
          </svg>
        </div>
        <p className="text-sm text-accent-red mb-4">
          Failed to generate notebook: {paper.notebook.error}
        </p>
        <Button
          variant="secondary"
          size="sm"
          onClick={() => {
            hasStartedRef.current = false;
            startStreaming({
              concept,
              paper: {
                title: paper.title,
                authors: paper.authors,
                year: paper.year,
                venue: paper.venue,
              },
            });
          }}
        >
          Retry
        </Button>
      </div>
    );
  }

  // Idle — waiting to start (brief flash before auto-start kicks in)
  return (
    <div className="text-center py-8">
      <div className="flex items-center justify-center gap-2 text-text-muted">
        <div className="flex gap-1">
          <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
          <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
          <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
        </div>
        <span className="text-sm">Preparing notebook...</span>
      </div>
    </div>
  );
}
