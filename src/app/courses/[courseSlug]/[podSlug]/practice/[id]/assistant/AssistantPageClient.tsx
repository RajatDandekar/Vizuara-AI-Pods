'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import Link from 'next/link';
import CourseProgressBar from '@/components/course/CourseProgressBar';
import ChatMessage, { type ChatMessageData } from '@/components/assistant/ChatMessage';
import ChatInput from '@/components/assistant/ChatInput';
import WelcomeCard from '@/components/assistant/WelcomeCard';
import { useChatStream } from '@/hooks/useChatStream';
import type { NotebookMeta } from '@/types/course';

interface Props {
  courseSlug: string;
  courseTitle: string;
  notebook: NotebookMeta;
  notebooks: NotebookMeta[];
  hasCaseStudy: boolean;
  notebookContext: string;
  toc: string[];
  podSlug: string;
}

const STORAGE_PREFIX = 'vizuara-chat-';

function getStorageKey(courseSlug: string, podSlug: string, order: number) {
  return `${STORAGE_PREFIX}${courseSlug}-${podSlug}-${order}`;
}

function loadMessages(key: string): ChatMessageData[] {
  try {
    const raw = localStorage.getItem(key);
    if (raw) return JSON.parse(raw);
  } catch { /* empty */ }
  return [];
}

function saveMessages(key: string, messages: ChatMessageData[]) {
  try {
    localStorage.setItem(key, JSON.stringify(messages));
  } catch { /* quota exceeded — ignore */ }
}

let nextId = 1;
function makeId() {
  return `msg-${Date.now()}-${nextId++}`;
}

export default function AssistantPageClient({
  courseSlug,
  courseTitle,
  notebook,
  notebooks,
  hasCaseStudy,
  notebookContext,
  podSlug,
}: Props) {
  const storageKey = getStorageKey(courseSlug, podSlug, notebook.order);
  const [messages, setMessages] = useState<ChatMessageData[]>([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const scrollRef = useRef<HTMLDivElement>(null);
  const autoScrollRef = useRef(true);
  const streamingIdRef = useRef<string | null>(null);

  // Load persisted messages on mount
  useEffect(() => {
    setMessages(loadMessages(storageKey));
  }, [storageKey]);

  // Save messages when they change (skip while streaming)
  useEffect(() => {
    if (!isStreaming && messages.length > 0) {
      saveMessages(storageKey, messages);
    }
  }, [messages, isStreaming, storageKey]);

  // Auto-scroll
  useEffect(() => {
    if (autoScrollRef.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  // Track scroll position for auto-scroll
  function handleScroll() {
    const el = scrollRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
    autoScrollRef.current = atBottom;
  }

  const onChunk = useCallback((text: string) => {
    setMessages((prev) => {
      const id = streamingIdRef.current;
      if (!id) return prev;
      return prev.map((m) =>
        m.id === id ? { ...m, content: m.content + text } : m
      );
    });
  }, []);

  const onDone = useCallback(() => {
    setIsStreaming(false);
    streamingIdRef.current = null;
  }, []);

  const onError = useCallback((message: string) => {
    setIsStreaming(false);
    setError(message);
    streamingIdRef.current = null;
  }, []);

  const { send, cancel } = useChatStream({ onChunk, onDone, onError });

  function handleSend(text?: string) {
    const question = (text || input).trim();
    if (!question || isStreaming) return;

    setError(null);
    setInput('');
    autoScrollRef.current = true;

    const userMsg: ChatMessageData = { id: makeId(), role: 'user', content: question };
    const assistantId = makeId();
    const assistantMsg: ChatMessageData = { id: assistantId, role: 'assistant', content: '' };
    streamingIdRef.current = assistantId;

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setIsStreaming(true);

    // Build history from prior messages (exclude the empty assistant stub)
    const history = [...messages, userMsg]
      .filter((m) => m.content.trim())
      .slice(-10)
      .map((m) => ({ role: m.role, content: m.content }));

    send({ question, context: notebookContext, history });
  }

  function handleStop() {
    cancel();
    setIsStreaming(false);
    streamingIdRef.current = null;
  }

  function handleClear() {
    cancel();
    setMessages([]);
    setIsStreaming(false);
    setError(null);
    streamingIdRef.current = null;
    try { localStorage.removeItem(storageKey); } catch { /* ignore */ }
  }

  const basePath = `/courses/${courseSlug}/${podSlug}`;

  return (
    <>
      <CourseProgressBar
        courseSlug={courseSlug}
        podSlug={podSlug}
        courseTitle={courseTitle}
        notebooks={notebooks}
        hasCaseStudy={hasCaseStudy}
        activeStep={`nb-${notebook.order}`}
      />

      <div className="flex flex-col h-[calc(100vh-5rem)]">
        {/* Header */}
        <div className="flex items-center justify-between px-4 sm:px-6 py-3 border-b border-card-border bg-white">
          <div className="flex items-center gap-3 min-w-0">
            <Link
              href={`${basePath}/practice/${notebook.order}`}
              className="flex-shrink-0 text-text-muted hover:text-foreground transition-colors"
              aria-label="Back to notebook"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 19.5L8.25 12l7.5-7.5" />
              </svg>
            </Link>
            <div className="min-w-0">
              <h1 className="text-sm font-semibold text-foreground truncate">AI Teaching Assistant</h1>
              <p className="text-xs text-text-muted truncate">{notebook.title}</p>
            </div>
          </div>
          {messages.length > 0 && (
            <button
              onClick={handleClear}
              className="text-xs text-text-muted hover:text-accent-red transition-colors flex-shrink-0 cursor-pointer"
            >
              Clear chat
            </button>
          )}
        </div>

        {/* Messages area */}
        <div
          ref={scrollRef}
          onScroll={handleScroll}
          className="flex-1 overflow-y-auto px-4 sm:px-6 py-4 space-y-3"
        >
          {messages.length === 0 ? (
            <WelcomeCard
              notebookTitle={notebook.title}
              onSelectQuestion={(q) => handleSend(q)}
            />
          ) : (
            <>
              {messages.map((msg) => (
                <ChatMessage
                  key={msg.id}
                  message={msg}
                  isStreaming={isStreaming && msg.id === streamingIdRef.current}
                />
              ))}
            </>
          )}

          {error && (
            <div className="text-center text-sm text-accent-red bg-red-50 rounded-xl px-4 py-2">
              {error}
            </div>
          )}
        </div>

        {/* Input */}
        <ChatInput
          value={input}
          onChange={setInput}
          onSend={() => handleSend()}
          onStop={handleStop}
          isStreaming={isStreaming}
        />
      </div>
    </>
  );
}
