'use client';

import { motion } from 'framer-motion';
import MarkdownRenderer from '@/components/ui/MarkdownRenderer';

export interface ChatMessageData {
  id: string;
  role: 'user' | 'assistant';
  content: string;
}

interface ChatMessageProps {
  message: ChatMessageData;
  isStreaming?: boolean;
}

export default function ChatMessage({ message, isStreaming }: ChatMessageProps) {
  const isUser = message.role === 'user';

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, ease: [0.25, 0.1, 0.25, 1] as [number, number, number, number] }}
      className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}
    >
      {isUser ? (
        <div className="max-w-[85%] px-4 py-2.5 rounded-2xl rounded-br-md bg-gradient-to-br from-accent-blue to-blue-700 text-white text-base leading-relaxed whitespace-pre-wrap">
          {message.content}
        </div>
      ) : (
        <div
          className={`max-w-[85%] bg-card-bg border border-card-border rounded-2xl rounded-bl-md px-4 py-3 chat-markdown ${isStreaming ? 'streaming-cursor' : ''}`}
        >
          <MarkdownRenderer content={message.content || '\u200B'} />
        </div>
      )}
    </motion.div>
  );
}
