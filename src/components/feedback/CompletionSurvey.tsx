'use client';

import { useState, useEffect } from 'react';
import { useAuth } from '@/context/AuthContext';
import { useFeedback } from '@/hooks/useFeedback';
import type { FeedbackSubmission } from '@/types/feedback';

const QUESTIONS = [
  {
    key: 'clarity',
    question: 'How clear were the explanations?',
    options: ['Very clear', 'Mostly clear', 'Somewhat unclear', 'Very confusing'],
  },
  {
    key: 'notebooks',
    question: 'How were the practice notebooks?',
    options: ['Excellent', 'Good', 'Could be better', 'Not helpful'],
  },
  {
    key: 'pace',
    question: 'How was the pace of the content?',
    options: ['Too fast', 'Just right', 'Too slow'],
  },
];

interface Props {
  courseSlug: string;
  podSlug?: string;
  contentType: 'pod' | 'course';
}

export default function CompletionSurvey({ courseSlug, podSlug, contentType }: Props) {
  const { user } = useAuth();
  const { submit, checkExisting, submitting } = useFeedback();
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [done, setDone] = useState(false);

  useEffect(() => {
    if (!user) return;
    checkExisting({ type: 'survey', courseSlug, podSlug, contentType }).then(existing => {
      if (existing?.surveyData) {
        setAnswers(existing.surveyData as Record<string, string>);
        setDone(true);
      }
    });
  }, [checkExisting, courseSlug, podSlug, contentType, user]);

  if (!user) return null;

  function handleAnswer(key: string, value: string) {
    setAnswers(prev => ({ ...prev, [key]: value }));
  }

  async function handleSubmit() {
    const data: FeedbackSubmission = {
      type: 'survey',
      courseSlug,
      podSlug,
      contentType,
      surveyData: answers,
    };
    await submit(data);
    setDone(true);
  }

  const allAnswered = QUESTIONS.every(q => answers[q.key]);

  return (
    <div className="border border-card-border rounded-xl p-5 my-4 bg-card-bg">
      <p className="text-sm font-medium text-foreground mb-4">
        Quick feedback survey
      </p>

      <div className="space-y-4">
        {QUESTIONS.map(({ key, question, options }) => (
          <div key={key}>
            <p className="text-sm text-text-secondary mb-2">{question}</p>
            <div className="flex flex-wrap gap-1.5">
              {options.map(option => (
                <button
                  key={option}
                  onClick={() => !done && handleAnswer(key, option)}
                  disabled={done}
                  className={`text-xs px-3 py-1.5 rounded-full border transition-colors cursor-pointer ${
                    answers[key] === option
                      ? 'bg-accent-blue-light border-accent-blue text-accent-blue'
                      : 'border-gray-200 text-text-muted hover:border-gray-300'
                  } ${done ? 'opacity-75' : ''}`}
                >
                  {option}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>

      {!done && (
        <button
          onClick={handleSubmit}
          disabled={submitting || !allAnswered}
          className="mt-4 text-xs font-medium px-4 py-2 rounded-lg bg-foreground text-white hover:bg-gray-800 transition-colors disabled:opacity-50 cursor-pointer"
        >
          Submit Survey
        </button>
      )}
      {done && (
        <p className="text-xs text-accent-green mt-3">Thanks for completing the survey!</p>
      )}
    </div>
  );
}
