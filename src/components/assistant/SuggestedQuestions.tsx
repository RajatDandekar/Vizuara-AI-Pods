'use client';

const DEFAULT_QUESTIONS = [
  'Explain the main concept',
  'Help with TODO exercises',
  'Key takeaways',
  'Explain the code step by step',
];

interface SuggestedQuestionsProps {
  onSelect: (question: string) => void;
}

export default function SuggestedQuestions({ onSelect }: SuggestedQuestionsProps) {
  return (
    <div className="flex flex-wrap gap-2">
      {DEFAULT_QUESTIONS.map((q) => (
        <button
          key={q}
          onClick={() => onSelect(q)}
          className="px-3.5 py-1.5 rounded-full text-xs font-medium border border-card-border text-text-secondary bg-white hover:border-accent-blue hover:text-accent-blue transition-colors cursor-pointer"
        >
          {q}
        </button>
      ))}
    </div>
  );
}
