'use client';

import MarkdownRenderer from './MarkdownRenderer';

interface StreamingTextProps {
  content: string;
  isStreaming: boolean;
}

export default function StreamingText({ content, isStreaming }: StreamingTextProps) {
  if (!content && isStreaming) {
    return (
      <div className="flex items-center gap-2 text-text-muted py-4">
        <div className="flex gap-1">
          <span className="w-1.5 h-1.5 bg-accent-blue rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
          <span className="w-1.5 h-1.5 bg-accent-blue rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
          <span className="w-1.5 h-1.5 bg-accent-blue rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
        </div>
        <span className="text-sm">Generating...</span>
      </div>
    );
  }

  return (
    <div className={isStreaming ? 'streaming-cursor' : ''}>
      <MarkdownRenderer content={content} />
    </div>
  );
}
