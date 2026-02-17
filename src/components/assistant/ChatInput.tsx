'use client';

import { useRef, useEffect, type KeyboardEvent } from 'react';

interface ChatInputProps {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  onStop: () => void;
  isStreaming: boolean;
  disabled?: boolean;
}

export default function ChatInput({
  value,
  onChange,
  onSend,
  onStop,
  isStreaming,
  disabled,
}: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 144) + 'px'; // 6 lines max
  }, [value]);

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isStreaming && value.trim()) onSend();
    }
  }

  return (
    <div className="flex items-end gap-2 p-3 border-t border-card-border bg-white">
      <textarea
        ref={textareaRef}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Ask about concepts, code, exercises..."
        disabled={disabled}
        rows={1}
        className="flex-1 resize-none rounded-xl border border-card-border px-4 py-2.5 text-sm leading-relaxed outline-none transition-colors focus:border-accent-blue disabled:opacity-50 bg-white"
      />
      {isStreaming ? (
        <button
          onClick={onStop}
          className="flex-shrink-0 w-10 h-10 rounded-full bg-accent-red text-white flex items-center justify-center transition-transform hover:scale-105 cursor-pointer"
          aria-label="Stop generating"
        >
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
            <rect x="6" y="6" width="12" height="12" rx="2" />
          </svg>
        </button>
      ) : (
        <button
          onClick={onSend}
          disabled={!value.trim() || disabled}
          className="flex-shrink-0 w-10 h-10 rounded-full bg-accent-blue text-white flex items-center justify-center transition-transform hover:scale-105 disabled:opacity-40 disabled:hover:scale-100 cursor-pointer disabled:cursor-not-allowed"
          aria-label="Send message"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
          </svg>
        </button>
      )}
    </div>
  );
}
