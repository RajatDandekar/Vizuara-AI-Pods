'use client';

import FadeIn from '@/components/animations/FadeIn';
import SuggestedQuestions from './SuggestedQuestions';

interface WelcomeCardProps {
  notebookTitle: string;
  onSelectQuestion: (question: string) => void;
}

export default function WelcomeCard({ notebookTitle, onSelectQuestion }: WelcomeCardProps) {
  return (
    <FadeIn className="flex flex-col items-center justify-center h-full px-4">
      <div className="max-w-md w-full text-center">
        <div className="w-14 h-14 rounded-2xl bg-accent-blue-light text-accent-blue flex items-center justify-center mx-auto mb-4">
          <svg className="w-7 h-7" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M8.625 12a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H8.25m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H12m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0h-.375M21 12c0 4.556-4.03 8.25-9 8.25a9.764 9.764 0 01-2.555-.337A5.972 5.972 0 015.41 20.97a5.969 5.969 0 01-.474-.065 4.48 4.48 0 00.978-2.025c.09-.457-.133-.901-.467-1.226C3.93 16.178 3 14.189 3 12c0-4.556 4.03-8.25 9-8.25s9 3.694 9 8.25z" />
          </svg>
        </div>
        <h2 className="text-lg font-semibold text-foreground mb-1">
          AI Teaching Assistant
        </h2>
        <p className="text-sm text-text-secondary mb-6 leading-relaxed">
          I&apos;ve read through <span className="font-medium text-foreground">{notebookTitle}</span>.
          Ask me about any concept, code block, or exercise.
        </p>
        <SuggestedQuestions onSelect={onSelectQuestion} />
      </div>
    </FadeIn>
  );
}
