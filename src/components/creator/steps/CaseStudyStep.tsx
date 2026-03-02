'use client';

import { useState, useCallback, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useCreator } from '@/context/CreatorContext';
import { useCreatorStream } from '@/hooks/useCreatorStream';
import StepHeader from '../shared/StepHeader';
import MarkdownRenderer from '@/components/ui/MarkdownRenderer';
import NotebookPreview from '../notebooks/NotebookPreview';
import FadeIn from '@/components/animations/FadeIn';

type Phase =
  | 'empty'           // No case study yet
  | 'generating'      // Streaming case study
  | 'review'          // Case study ready, user reviews
  | 'gen-notebook'    // Streaming notebook markdown
  | 'notebook-ready'  // Notebook markdown done
  | 'converting'      // Converting to .ipynb
  | 'converted'       // .ipynb ready
  | 'uploading'       // Uploading to Drive
  | 'complete';       // Everything done — Colab URL available

function derivePhase(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  caseStudy: any,
  streamingStatus: string
): Phase {
  if (!caseStudy?.content && streamingStatus !== 'streaming') return 'empty';
  if (streamingStatus === 'streaming') {
    // Determine if we're streaming the case study or notebook
    if (caseStudy?.content && !caseStudy?.notebookMarkdown) return 'gen-notebook';
    if (!caseStudy?.content) return 'generating';
    return 'generating';
  }
  if (caseStudy?.notebookColabUrl) return 'complete';
  if (caseStudy?.notebookIpynbStoragePath) return 'converted';
  if (caseStudy?.notebookStatus === 'converting') return 'converting';
  if (caseStudy?.notebookMarkdown) return 'notebook-ready';
  if (caseStudy?.content) return 'review';
  return 'empty';
}

const PHASE_STEPS = [
  { key: 'case-study', label: 'Case Study' },
  { key: 'notebook', label: 'Colab Notebook' },
  { key: 'convert', label: 'Convert .ipynb' },
  { key: 'upload', label: 'Upload to Drive' },
  { key: 'pdf', label: 'PDF (Optional)' },
];

function getCompletedSteps(phase: Phase): number {
  switch (phase) {
    case 'empty':
    case 'generating':
      return 0;
    case 'review':
      return 1;
    case 'gen-notebook':
    case 'notebook-ready':
      return 2;
    case 'converting':
    case 'converted':
      return 3;
    case 'uploading':
      return 3;
    case 'complete':
      return 4;
    default:
      return 0;
  }
}

export default function CaseStudyStep() {
  const router = useRouter();
  const { state, dispatch, refreshProject } = useCreator();
  const project = state.activeProject;

  const [phase, setPhase] = useState<Phase>('empty');
  const [pdfLoading, setPdfLoading] = useState(false);
  const [pdfDone, setPdfDone] = useState(false);
  const [convertLoading, setConvertLoading] = useState(false);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [notebookStreamContent, setNotebookStreamContent] = useState('');
  const [activeStream, setActiveStream] = useState<'case-study' | 'notebook' | null>(null);
  const [showNotebookPreview, setShowNotebookPreview] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Keep phase in sync with project state
  useEffect(() => {
    if (project?.caseStudy) {
      const derived = derivePhase(project.caseStudy, state.streamingStatus);
      setPhase(derived);
      if (project.caseStudy.pdfStoragePath) setPdfDone(true);
    }
  }, [project?.caseStudy, state.streamingStatus]);

  const caseStudyStream = useCreatorStream({
    onComplete: useCallback(() => {
      if (project) {
        setActiveStream(null);
        refreshProject(project.id);
      }
    }, [project, refreshProject]),
  });

  const notebookStream = useCreatorStream({
    onComplete: useCallback(() => {
      if (project) {
        setActiveStream(null);
        setNotebookStreamContent('');
        refreshProject(project.id);
      }
    }, [project, refreshProject]),
  });

  if (!project) return null;

  const caseStudy = project.caseStudy;
  const completedSteps = getCompletedSteps(phase);

  // --- Handlers ---

  const handleGenerate = () => {
    setError(null);
    setActiveStream('case-study');
    dispatch({ type: 'STREAM_RESET' });
    caseStudyStream.startStream('/api/admin/creator/case-study/generate', {
      projectId: project.id,
    });
  };

  const handleGenerateNotebook = () => {
    setError(null);
    setActiveStream('notebook');
    setNotebookStreamContent('');
    dispatch({ type: 'STREAM_RESET' });
    notebookStream.startStream('/api/admin/creator/case-study/notebook', {
      projectId: project.id,
    });
  };

  const handleConvert = async () => {
    setError(null);
    setConvertLoading(true);
    setPhase('converting');
    try {
      const res = await fetch('/api/admin/creator/case-study/convert', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ projectId: project.id }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Convert failed');
      await refreshProject(project.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Conversion failed');
      setPhase('notebook-ready');
    } finally {
      setConvertLoading(false);
    }
  };

  const handleUpload = async () => {
    setError(null);
    setUploadLoading(true);
    setPhase('uploading');
    try {
      const res = await fetch('/api/admin/creator/case-study/upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ projectId: project.id }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Upload failed');
      await refreshProject(project.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
      setPhase('converted');
    } finally {
      setUploadLoading(false);
    }
  };

  const handleGeneratePdf = async () => {
    setError(null);
    setPdfLoading(true);
    try {
      const res = await fetch('/api/admin/creator/case-study/pdf', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ projectId: project.id }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'PDF generation failed');
      setPdfDone(true);
      await refreshProject(project.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'PDF generation failed');
    } finally {
      setPdfLoading(false);
    }
  };

  const handleProceed = async () => {
    await fetch(`/api/admin/creator/projects/${project.id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ currentStep: 'narration' }),
    });
    dispatch({
      type: 'UPDATE_PROJECT',
      project: { ...project, currentStep: 'narration' },
    });
    router.push(`/admin/creator/${project.id}/narration`);
  };

  const handleReset = async () => {
    setError(null);
    setPhase('empty');
    setNotebookStreamContent('');
    setPdfDone(false);
    setShowNotebookPreview(false);
    dispatch({ type: 'STREAM_RESET' });
    await fetch(`/api/admin/creator/projects/${project.id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ caseStudy: undefined }),
    });
    await refreshProject(project.id);
  };

  // Determine what's streaming
  const isStreamingCaseStudy = activeStream === 'case-study' && state.streamingStatus === 'streaming';
  const isStreamingNotebook = activeStream === 'notebook' && state.streamingStatus === 'streaming';

  // Track notebook stream content
  useEffect(() => {
    if (isStreamingNotebook) {
      setNotebookStreamContent(state.streamingContent);
    }
  }, [isStreamingNotebook, state.streamingContent]);

  return (
    <div className="p-8">
      <StepHeader
        title="Case Study"
        description="Generate a real-world case study with industry context, technical formulation, Colab notebook, and system design"
        stepNumber={6}
      />

      {/* Progress bar */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-2">
          {PHASE_STEPS.map((step, i) => (
            <div key={step.key} className="flex items-center gap-2">
              <div
                className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold
                  ${i < completedSteps
                    ? 'bg-accent-green text-white'
                    : i === completedSteps
                      ? 'bg-accent-blue text-white'
                      : 'bg-card-bg border border-card-border text-text-muted'
                  }`}
              >
                {i < completedSteps ? (
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                ) : (
                  i + 1
                )}
              </div>
              <span className={`text-sm font-medium hidden sm:inline ${
                i <= completedSteps ? 'text-foreground' : 'text-text-muted'
              }`}>
                {step.label}
              </span>
              {i < PHASE_STEPS.length - 1 && (
                <div className={`w-8 h-0.5 mx-1 ${
                  i < completedSteps ? 'bg-accent-green' : 'bg-card-border'
                }`} />
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Error banner */}
      {error && (
        <div className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-base text-red-400">
          {error}
        </div>
      )}

      <FadeIn>
        {/* Phase: Empty — Generate button */}
        {phase === 'empty' && !isStreamingCaseStudy && (
          <div className="text-center py-16 border border-dashed border-card-border rounded-xl">
            <div className="w-16 h-16 rounded-full bg-accent-blue/10 flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-accent-blue" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                  d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
              </svg>
            </div>
            <h3 className="text-lg font-semibold text-foreground mb-2">Generate Case Study</h3>
            <p className="text-text-muted text-base mb-6 max-w-md mx-auto">
              Claude will create a 4-section case study: Industry Context, Technical Formulation,
              Implementation Notebook, and Production System Design.
            </p>
            <button
              onClick={handleGenerate}
              className="px-6 py-3 bg-accent-blue text-white rounded-lg font-medium
                         hover:bg-accent-blue/90 transition-colors"
            >
              Generate Case Study
            </button>
          </div>
        )}

        {/* Phase: Streaming case study */}
        {(isStreamingCaseStudy || phase === 'generating') && (
          <div className="space-y-4">
            <div className="flex items-center gap-2 text-base text-accent-blue">
              <div className="w-4 h-4 border-2 border-accent-blue border-t-transparent rounded-full animate-spin" />
              Generating 4-section case study...
            </div>
            <div className="border border-card-border rounded-xl p-6 max-h-[600px] overflow-auto bg-card-bg streaming-cursor">
              <MarkdownRenderer
                content={state.streamingContent}
                streaming={true}
              />
            </div>
          </div>
        )}

        {/* Phase: Review case study */}
        {phase === 'review' && !isStreamingCaseStudy && (
          <div className="space-y-4">
            {/* Case study preview - collapsible */}
            <div className="border border-card-border rounded-xl overflow-hidden">
              <button
                onClick={() => setShowNotebookPreview(!showNotebookPreview)}
                className="w-full px-6 py-4 bg-card-bg flex items-center justify-between hover:bg-card-bg/80 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-full bg-accent-green/10 flex items-center justify-center">
                    <svg className="w-5 h-5 text-accent-green" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  </div>
                  <div className="text-left">
                    <p className="font-medium text-foreground">Case Study Document</p>
                    <p className="text-sm text-text-muted">4-section case study with industry context and technical formulation</p>
                  </div>
                </div>
                <svg className={`w-5 h-5 text-text-muted transition-transform ${showNotebookPreview ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>
              {showNotebookPreview && (
                <div className="border-t border-card-border p-6 max-h-[500px] overflow-auto">
                  <MarkdownRenderer content={caseStudy?.content || ''} />
                </div>
              )}
            </div>

            {/* Action: Generate notebook */}
            <div className="border border-accent-blue/30 rounded-xl p-6 bg-accent-blue/5">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="font-semibold text-foreground">Generate Colab Notebook</h3>
                  <p className="text-base text-text-muted mt-1">
                    Create an interactive notebook from Section 3 with TODO exercises and verification cells
                  </p>
                </div>
                <button
                  onClick={handleGenerateNotebook}
                  className="px-5 py-2.5 bg-accent-blue text-white rounded-lg font-medium
                             hover:bg-accent-blue/90 transition-colors whitespace-nowrap"
                >
                  Generate Notebook
                </button>
              </div>
            </div>

            {/* Secondary actions */}
            <div className="flex items-center gap-3">
              <button
                onClick={handleGenerate}
                className="px-4 py-2 text-base border border-card-border rounded-lg text-text-secondary
                           hover:bg-background transition-colors"
              >
                Regenerate Case Study
              </button>
              <button
                onClick={handleReset}
                className="px-4 py-2 text-base text-text-muted hover:text-red-400 transition-colors"
              >
                Reset
              </button>
            </div>
          </div>
        )}

        {/* Phase: Streaming notebook */}
        {isStreamingNotebook && (
          <div className="space-y-4">
            <CaseStudySummaryCard content={caseStudy?.content} />
            <div className="flex items-center gap-2 text-sm text-accent-blue">
              <div className="w-4 h-4 border-2 border-accent-blue border-t-transparent rounded-full animate-spin" />
              Generating Colab notebook from Section 3...
            </div>
            <div className="border border-card-border rounded-xl p-6 max-h-[500px] overflow-auto bg-card-bg streaming-cursor">
              <MarkdownRenderer
                content={notebookStreamContent || state.streamingContent}
                streaming={true}
              />
            </div>
          </div>
        )}

        {/* Phase: Notebook ready — show preview + convert button */}
        {(phase === 'notebook-ready' || phase === 'converting') && (
          <div className="space-y-4">
            <CaseStudySummaryCard content={caseStudy?.content} />

            {/* Notebook preview */}
            <div className="border border-accent-green/30 rounded-xl overflow-hidden">
              <div className="px-6 py-4 bg-accent-green/5 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-full bg-accent-green/10 flex items-center justify-center">
                    <svg className="w-5 h-5 text-accent-green" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  </div>
                  <div>
                    <p className="font-medium text-foreground">Notebook Generated</p>
                    <p className="text-sm text-text-muted">Implementation notebook with TODO scaffolding</p>
                  </div>
                </div>
                <button
                  onClick={handleConvert}
                  disabled={convertLoading || phase === 'converting'}
                  className="px-5 py-2.5 bg-accent-blue text-white rounded-lg font-medium
                             hover:bg-accent-blue/90 transition-colors disabled:opacity-50 whitespace-nowrap"
                >
                  {convertLoading || phase === 'converting' ? (
                    <span className="flex items-center gap-2">
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                      Converting...
                    </span>
                  ) : (
                    'Convert to .ipynb'
                  )}
                </button>
              </div>
              <div className="border-t border-card-border p-4 max-h-[400px] overflow-auto">
                <NotebookPreview
                  content={caseStudy?.notebookMarkdown || ''}
                  title="Case Study Notebook"
                />
              </div>
            </div>

            <div className="flex items-center gap-3">
              <button
                onClick={handleGenerateNotebook}
                className="px-4 py-2 text-base border border-card-border rounded-lg text-text-secondary
                           hover:bg-background transition-colors"
              >
                Regenerate Notebook
              </button>
            </div>
          </div>
        )}

        {/* Phase: Converted — upload button */}
        {(phase === 'converted' || phase === 'uploading') && (
          <div className="space-y-4">
            <CaseStudySummaryCard content={caseStudy?.content} />

            <div className="border border-accent-green/30 rounded-xl p-6 bg-accent-green/5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-full bg-accent-green/10 flex items-center justify-center">
                    <svg className="w-5 h-5 text-accent-green" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  </div>
                  <div>
                    <p className="font-medium text-foreground">Notebook Converted</p>
                    <p className="text-sm text-text-muted">.ipynb file ready for upload</p>
                  </div>
                </div>
                <button
                  onClick={handleUpload}
                  disabled={uploadLoading || phase === 'uploading'}
                  className="px-5 py-2.5 bg-accent-blue text-white rounded-lg font-medium
                             hover:bg-accent-blue/90 transition-colors disabled:opacity-50 whitespace-nowrap"
                >
                  {uploadLoading || phase === 'uploading' ? (
                    <span className="flex items-center gap-2">
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                      Uploading...
                    </span>
                  ) : (
                    'Upload to Drive'
                  )}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Phase: Complete — show Colab URL + PDF option + proceed */}
        {phase === 'complete' && (
          <div className="space-y-4">
            <CaseStudySummaryCard content={caseStudy?.content} />

            {/* Colab link */}
            <div className="border border-accent-green/30 rounded-xl p-6 bg-accent-green/5">
              <div className="flex items-center gap-3 mb-3">
                <div className="w-8 h-8 rounded-full bg-accent-green/10 flex items-center justify-center">
                  <svg className="w-5 h-5 text-accent-green" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <div>
                  <p className="font-semibold text-foreground">Case Study Notebook Ready</p>
                  <p className="text-sm text-text-muted">Uploaded to Google Drive and linked to Colab</p>
                </div>
              </div>
              <a
                href={caseStudy?.notebookColabUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-2.5 bg-white/10 border border-accent-green/30
                           rounded-lg text-base font-medium text-accent-green hover:bg-accent-green/10 transition-colors"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                </svg>
                Open in Google Colab
              </a>
            </div>

            {/* PDF generation */}
            <div className="border border-card-border rounded-xl p-6 bg-card-bg">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium text-foreground">PDF Export</p>
                  <p className="text-sm text-text-muted">Generate a styled PDF of the case study document</p>
                </div>
                {pdfDone ? (
                  <span className="text-base text-accent-green font-medium flex items-center gap-1.5">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                    PDF Generated
                  </span>
                ) : (
                  <button
                    onClick={handleGeneratePdf}
                    disabled={pdfLoading}
                    className="px-4 py-2 text-sm border border-card-border rounded-lg
                               hover:bg-background transition-colors disabled:opacity-50"
                  >
                    {pdfLoading ? (
                      <span className="flex items-center gap-2">
                        <div className="w-3 h-3 border-2 border-text-muted border-t-transparent rounded-full animate-spin" />
                        Generating...
                      </span>
                    ) : (
                      'Generate PDF'
                    )}
                  </button>
                )}
              </div>
            </div>

            {/* Proceed */}
            <div className="flex items-center gap-3 pt-2">
              <button
                onClick={handleProceed}
                className="px-6 py-3 bg-accent-green text-white rounded-lg font-medium
                           hover:bg-accent-green/90 transition-colors"
              >
                Continue to Narration
              </button>
              <button
                onClick={handleReset}
                className="px-4 py-2 text-base text-text-muted hover:text-red-400 transition-colors"
              >
                Start Over
              </button>
            </div>
          </div>
        )}
      </FadeIn>
    </div>
  );
}

/**
 * Collapsed summary card showing the case study doc is ready
 */
function CaseStudySummaryCard({ content }: { content?: string }) {
  const [expanded, setExpanded] = useState(false);

  if (!content) return null;

  // Extract section headings for a quick summary
  const headings = content.match(/^## Section \d:.+$/gm) || [];

  return (
    <div className="border border-card-border rounded-xl overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-6 py-3 bg-card-bg flex items-center justify-between hover:bg-card-bg/80 transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className="w-6 h-6 rounded-full bg-accent-green/10 flex items-center justify-center">
            <svg className="w-4 h-4 text-accent-green" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          </div>
          <div className="text-left">
            <p className="text-base font-medium text-foreground">Case Study Document</p>
            <p className="text-sm text-text-muted">
              {headings.length > 0
                ? headings.map(h => h.replace(/^## Section \d: /, '')).join(' / ')
                : '4-section case study'}
            </p>
          </div>
        </div>
        <svg className={`w-4 h-4 text-text-muted transition-transform ${expanded ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      {expanded && (
        <div className="border-t border-card-border p-6 max-h-[400px] overflow-auto">
          <MarkdownRenderer content={content} />
        </div>
      )}
    </div>
  );
}
