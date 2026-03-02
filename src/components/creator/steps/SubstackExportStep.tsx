'use client';

import { useState, useCallback, useMemo, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { useCreator } from '@/context/CreatorContext';
import StepHeader from '../shared/StepHeader';
import MarkdownRenderer from '@/components/ui/MarkdownRenderer';
import FadeIn from '@/components/animations/FadeIn';
import type { FigureState } from '@/types/creator';

interface BlockEquation {
  index: number;
  latex: string;
}

/**
 * Convert LaTeX to readable Unicode text for inline math.
 * Handles Greek letters, super/subscripts, common operators.
 */
function latexToUnicode(latex: string): string {
  let t = latex;

  // Greek letters
  const greek: Record<string, string> = {
    '\\alpha': '\u03B1', '\\beta': '\u03B2', '\\gamma': '\u03B3', '\\delta': '\u03B4',
    '\\epsilon': '\u03B5', '\\varepsilon': '\u03B5', '\\zeta': '\u03B6', '\\eta': '\u03B7',
    '\\theta': '\u03B8', '\\iota': '\u03B9', '\\kappa': '\u03BA', '\\lambda': '\u03BB',
    '\\mu': '\u03BC', '\\nu': '\u03BD', '\\xi': '\u03BE', '\\pi': '\u03C0',
    '\\rho': '\u03C1', '\\sigma': '\u03C3', '\\tau': '\u03C4', '\\phi': '\u03C6',
    '\\varphi': '\u03C6', '\\chi': '\u03C7', '\\psi': '\u03C8', '\\omega': '\u03C9',
    '\\Gamma': '\u0393', '\\Delta': '\u0394', '\\Theta': '\u0398', '\\Lambda': '\u039B',
    '\\Xi': '\u039E', '\\Pi': '\u03A0', '\\Sigma': '\u03A3', '\\Phi': '\u03A6',
    '\\Psi': '\u03A8', '\\Omega': '\u03A9',
  };
  for (const [cmd, ch] of Object.entries(greek)) {
    t = t.replace(new RegExp(cmd.replace(/\\/g, '\\\\') + '(?![a-zA-Z])', 'g'), ch);
  }

  // Operators and symbols
  const ops: [RegExp, string][] = [
    [/\\cdot/g, '\u00B7'], [/\\times/g, '\u00D7'], [/\\div/g, '\u00F7'],
    [/\\pm/g, '\u00B1'], [/\\mp/g, '\u2213'], [/\\leq/g, '\u2264'], [/\\geq/g, '\u2265'],
    [/\\neq/g, '\u2260'], [/\\approx/g, '\u2248'], [/\\equiv/g, '\u2261'],
    [/\\infty/g, '\u221E'], [/\\partial/g, '\u2202'], [/\\nabla/g, '\u2207'],
    [/\\sum/g, '\u2211'], [/\\prod/g, '\u220F'], [/\\int/g, '\u222B'],
    [/\\rightarrow/g, '\u2192'], [/\\leftarrow/g, '\u2190'], [/\\Rightarrow/g, '\u21D2'],
    [/\\Leftarrow/g, '\u21D0'], [/\\in/g, '\u2208'], [/\\notin/g, '\u2209'],
    [/\\subset/g, '\u2282'], [/\\supset/g, '\u2283'], [/\\forall/g, '\u2200'],
    [/\\exists/g, '\u2203'], [/\\ldots/g, '\u2026'], [/\\cdots/g, '\u22EF'],
    [/\\vdots/g, '\u22EE'], [/\\ddots/g, '\u22F1'],
    [/\\langle/g, '\u27E8'], [/\\rangle/g, '\u27E9'],
  ];
  for (const [re, ch] of ops) t = t.replace(re, ch);

  // Superscripts
  const sup: Record<string, string> = {
    '0': '\u2070', '1': '\u00B9', '2': '\u00B2', '3': '\u00B3', '4': '\u2074',
    '5': '\u2075', '6': '\u2076', '7': '\u2077', '8': '\u2078', '9': '\u2079',
    'n': '\u207F', 'i': '\u2071', '+': '\u207A', '-': '\u207B', 'T': '\u1D40',
  };
  const sub: Record<string, string> = {
    '0': '\u2080', '1': '\u2081', '2': '\u2082', '3': '\u2083', '4': '\u2084',
    '5': '\u2085', '6': '\u2086', '7': '\u2087', '8': '\u2088', '9': '\u2089',
    'i': '\u1D62', 'j': '\u2C7C', 'k': '\u2096', 'n': '\u2099',
  };
  t = t.replace(/\^{([^}]*)}/g, (_, c) => c.split('').map((x: string) => sup[x] || x).join(''));
  t = t.replace(/\^([0-9a-zA-Z])/g, (_, c) => sup[c] || `^${c}`);
  t = t.replace(/_{([^}]*)}/g, (_, c) => c.split('').map((x: string) => sub[x] || x).join(''));
  t = t.replace(/_([0-9a-zA-Z])/g, (_, c) => sub[c] || `_${c}`);

  // Common constructs
  t = t.replace(/\\sqrt{([^}]*)}/g, '\u221A($1)');
  t = t.replace(/\\frac{([^}]*)}{([^}]*)}/g, '($1/$2)');
  t = t.replace(/\\mathbf{([^}]*)}/g, '$1');
  t = t.replace(/\\mathrm{([^}]*)}/g, '$1');
  t = t.replace(/\\mathcal{([^}]*)}/g, '$1');
  t = t.replace(/\\text{([^}]*)}/g, '$1');
  t = t.replace(/\\operatorname{([^}]*)}/g, '$1');
  t = t.replace(/\\left/g, '');
  t = t.replace(/\\right/g, '');
  t = t.replace(/\\,/g, ' ');
  t = t.replace(/\\;/g, ' ');
  t = t.replace(/\\!/g, '');
  t = t.replace(/\\ /g, ' ');
  t = t.replace(/\\quad/g, '  ');
  t = t.replace(/\\qquad/g, '    ');

  // Remove remaining LaTeX commands
  t = t.replace(/\\[a-zA-Z]+/g, '');
  // Clean braces and extra whitespace
  t = t.replace(/[{}]/g, '');
  t = t.replace(/\s+/g, ' ').trim();

  return t;
}

/**
 * Extract block equations ($$...$$) from markdown and replace with numbered placeholders.
 * Handles all formats: $$ on own line, $$content$$, $$content\n...\ncontent$$
 */
function extractEquations(markdown: string): {
  content: string;
  equations: BlockEquation[];
} {
  const equations: BlockEquation[] = [];
  let index = 1;
  let inCodeBlock = false;

  const lines = markdown.split('\n');
  const result: string[] = [];
  let inMath = false;
  let mathLines: string[] = [];

  const pushEquation = (latex: string) => {
    if (latex) {
      equations.push({ index, latex });
      result.push('');
      result.push(`**[Equation ${index}** — see Equations tab below]`);
      result.push('');
      index++;
    }
  };

  for (const line of lines) {
    if (line.trimStart().startsWith('```')) {
      inCodeBlock = !inCodeBlock;
      result.push(line);
      continue;
    }
    if (inCodeBlock) {
      result.push(line);
      continue;
    }

    const trimmed = line.trim();

    // Already in a math block — look for closing
    if (inMath) {
      if (trimmed === '$$') {
        // Closing $$ on its own line
        pushEquation(mathLines.join('\n').trim());
        inMath = false;
      } else if (trimmed.endsWith('$$')) {
        // Closing $$ at end of a content line
        mathLines.push(trimmed.slice(0, -2));
        pushEquation(mathLines.join('\n').trim());
        inMath = false;
      } else {
        mathLines.push(line);
      }
      continue;
    }

    // Not in math block — check for opening patterns

    // Case 1: Line is exactly "$$" — opens a multi-line block
    if (trimmed === '$$') {
      inMath = true;
      mathLines = [];
      continue;
    }

    // Case 2: Single-line $$...$$
    if (trimmed.startsWith('$$') && trimmed.endsWith('$$') && trimmed.length > 4) {
      pushEquation(trimmed.slice(2, -2).trim());
      continue;
    }

    // Case 3: Opens with $$ + content, doesn't close — start multi-line block
    if (trimmed.startsWith('$$') && !trimmed.endsWith('$$')) {
      inMath = true;
      mathLines = [trimmed.slice(2)];
      continue;
    }

    result.push(line);
  }

  return { content: result.join('\n'), equations };
}

export default function SubstackExportStep() {
  const router = useRouter();
  const { state, dispatch } = useCreator();
  const project = state.activeProject;
  const contentRef = useRef<HTMLDivElement>(null);
  const [activeTab, setActiveTab] = useState<'preview' | 'equations'>('preview');
  const [copied, setCopied] = useState(false);
  const [copying, setCopying] = useState(false);
  const [eqCopied, setEqCopied] = useState<number | null>(null);

  const { processedContent, equations } = useMemo(() => {
    if (!project?.articleDraft) return { processedContent: '', equations: [] };

    let content = project.articleDraft;

    // Replace figure placeholders with actual images
    let figureIndex = 0;
    content = content.replace(
      /\{\{FIGURE:\s*(.+?)\s*(?:\|\|\s*(.+?))?\s*\}\}/g,
      (_match, _desc, caption) => {
        const figure = project.figures[figureIndex];
        figureIndex++;

        if (!figure) return '*[Figure missing]*';

        if (figure.storagePath) {
          const src = `/api/admin/creator/figures/serve?projectId=${project.id}&figureId=${figure.id}`;
          return `![${caption || figure.caption}](${src})`;
        }
        return `*[${figure.caption} — not yet generated]*`;
      }
    );

    // Extract equations and replace with placeholders
    const { content: cleaned, equations: eqs } = extractEquations(content);

    return { processedContent: cleaned, equations: eqs };
  }, [project]);

  const copyRichText = useCallback(async () => {
    if (!contentRef.current) return;
    setCopying(true);

    try {
      // Clone the DOM so we can modify it without affecting the preview
      const clone = contentRef.current.cloneNode(true) as HTMLElement;

      // 1. Replace KaTeX math with readable Unicode text
      //    KaTeX renders: <span class="katex"><span class="katex-mathml"><math>...<annotation encoding="application/x-tex">LATEX</annotation></math></span><span class="katex-html">...</span></span>
      //    We extract the LaTeX source and convert to Unicode approximation
      //    Block (display) math should already be extracted by extractEquations, but handle any remaining
      const katexDisplays = clone.querySelectorAll('.katex-display');
      katexDisplays.forEach((el) => {
        const annotation = el.querySelector('annotation[encoding="application/x-tex"]');
        if (annotation) {
          const latex = annotation.textContent || '';
          const text = document.createTextNode(`[Equation: ${latexToUnicode(latex)}]`);
          el.replaceWith(text);
        }
      });

      const katexInline = clone.querySelectorAll('.katex');
      katexInline.forEach((el) => {
        const annotation = el.querySelector('annotation[encoding="application/x-tex"]');
        if (annotation) {
          const latex = annotation.textContent || '';
          const unicode = latexToUnicode(latex);
          // Wrap in italic for single variables/short expressions
          const em = document.createElement('em');
          em.textContent = unicode;
          el.replaceWith(em);
        }
      });

      // 2. Convert local images to base64 data URIs so they embed when pasted
      const images = clone.querySelectorAll('img');
      await Promise.all(
        Array.from(images).map(async (img) => {
          try {
            const src = img.getAttribute('src');
            if (!src || src.startsWith('data:')) return;
            const response = await fetch(src);
            const blob = await response.blob();
            const dataUri = await new Promise<string>((resolve) => {
              const reader = new FileReader();
              reader.onloadend = () => resolve(reader.result as string);
              reader.readAsDataURL(blob);
            });
            img.setAttribute('src', dataUri);
          } catch {
            // Leave as-is if fetch fails
          }
        })
      );

      // 3. Copy as both rich text and plain text
      const html = clone.innerHTML;
      // textContent works on detached nodes (innerText doesn't)
      const text = clone.textContent || '';
      const htmlBlob = new Blob([html], { type: 'text/html' });
      const textBlob = new Blob([text], { type: 'text/plain' });

      await navigator.clipboard.write([
        new ClipboardItem({
          'text/html': htmlBlob,
          'text/plain': textBlob,
        }),
      ]);

      setCopied(true);
      setTimeout(() => setCopied(false), 2500);
    } catch {
      // Fallback: select and copy from original (won't fix images/math but at least copies something)
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(contentRef.current);
      selection?.removeAllRanges();
      selection?.addRange(range);
      document.execCommand('copy');
      selection?.removeAllRanges();

      setCopied(true);
      setTimeout(() => setCopied(false), 2500);
    } finally {
      setCopying(false);
    }
  }, []);

  const copyEquation = useCallback(async (index: number, latex: string) => {
    await navigator.clipboard.writeText(latex);
    setEqCopied(index);
    setTimeout(() => setEqCopied(null), 2000);
  }, []);

  if (!project) return null;

  const handleProceed = async () => {
    await fetch(`/api/admin/creator/projects/${project.id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ currentStep: 'notebooks' }),
    });
    dispatch({
      type: 'UPDATE_PROJECT',
      project: { ...project, currentStep: 'notebooks' },
    });
    router.push(`/admin/creator/${project.id}/notebooks`);
  };

  return (
    <div className="p-8">
      <StepHeader
        title="Substack Export"
        description="Copy the rendered article as rich text and paste directly into Substack"
        stepNumber={4}
      />

      <FadeIn>
        {/* Tabs */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex gap-1 bg-background p-1 rounded-lg">
            <button
              onClick={() => setActiveTab('preview')}
              className={`px-4 py-2 rounded-md text-base font-medium transition-colors
                ${activeTab === 'preview' ? 'bg-card-bg text-foreground shadow-sm' : 'text-text-muted hover:text-text-secondary'}`}
            >
              Article
            </button>
            <button
              onClick={() => setActiveTab('equations')}
              className={`px-4 py-2 rounded-md text-base font-medium transition-colors
                ${activeTab === 'equations' ? 'bg-card-bg text-foreground shadow-sm' : 'text-text-muted hover:text-text-secondary'}`}
            >
              Equations ({equations.length})
            </button>
          </div>

          {activeTab === 'preview' && (
            <button
              onClick={copyRichText}
              disabled={copying}
              className={`px-5 py-2 rounded-lg font-medium text-base transition-colors
                ${copied
                  ? 'bg-accent-green text-white'
                  : copying
                    ? 'bg-accent-blue/60 text-white cursor-wait'
                    : 'bg-accent-blue text-white hover:bg-accent-blue/90'}`}
            >
              {copied ? 'Copied to Clipboard!' : copying ? 'Preparing...' : 'Copy Article as Rich Text'}
            </button>
          )}
        </div>

        {activeTab === 'preview' && (
          <div className="space-y-4">
            {/* Instructions */}
            <div className="p-3 bg-accent-blue/5 border border-accent-blue/20 rounded-lg">
              <p className="text-base text-text-secondary">
                <strong>How to use:</strong> Click &ldquo;Copy Article as Rich Text&rdquo; above, then paste directly
                into Substack&apos;s editor. Formatting, images, and code blocks will transfer.
                Images are embedded so they paste correctly.
                Inline math (like variables and symbols) is converted to Unicode text.
                For block equations, switch to the Equations tab and add each one using Substack&apos;s
                LaTeX block feature (+ button → Equation).
              </p>
            </div>

            {/* Rendered article */}
            <div
              ref={contentRef}
              className="border border-card-border rounded-xl p-8 bg-white"
            >
              <MarkdownRenderer content={processedContent} />
            </div>
          </div>
        )}

        {activeTab === 'equations' && (
          <div className="space-y-3">
            <div className="p-3 bg-accent-blue/5 border border-accent-blue/20 rounded-lg">
              <p className="text-base text-text-secondary">
                <strong>How to use:</strong> In Substack&apos;s editor, click the + button and select
                &ldquo;Equation&rdquo; to insert a LaTeX block. Copy each equation below and paste
                it into the corresponding placeholder in your article.
              </p>
            </div>

            {equations.length === 0 ? (
              <div className="text-center py-12 border border-dashed border-card-border rounded-xl">
                <p className="text-text-muted text-base">No block equations found in the article</p>
              </div>
            ) : (
              equations.map((eq) => (
                <div
                  key={eq.index}
                  className="border border-card-border rounded-xl bg-card-bg overflow-hidden"
                >
                  <div className="flex items-center justify-between px-4 py-2 bg-background border-b border-card-border">
                    <span className="text-base font-semibold text-foreground">
                      Equation {eq.index}
                    </span>
                    <button
                      onClick={() => copyEquation(eq.index, eq.latex)}
                      className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors
                        ${eqCopied === eq.index
                          ? 'bg-accent-green text-white'
                          : 'bg-accent-blue text-white hover:bg-accent-blue/90'}`}
                    >
                      {eqCopied === eq.index ? 'Copied!' : 'Copy LaTeX'}
                    </button>
                  </div>
                  <pre className="p-4 text-base text-foreground font-mono whitespace-pre-wrap break-all">
                    {eq.latex}
                  </pre>
                </div>
              ))
            )}
          </div>
        )}

        {/* Proceed */}
        <div className="flex justify-end mt-6">
          <button
            onClick={handleProceed}
            className="px-5 py-2.5 bg-accent-blue text-white rounded-lg font-medium
                       hover:bg-accent-blue/90 transition-colors"
          >
            Continue to Notebooks
          </button>
        </div>
      </FadeIn>
    </div>
  );
}
