'use client';

import { SUGGESTED_CONCEPTS } from '@/lib/constants';

interface ConceptSuggestionsProps {
  onSelect: (concept: string) => void;
}

export default function ConceptSuggestions({ onSelect }: ConceptSuggestionsProps) {
  return (
    <div className="flex flex-wrap gap-2 justify-center">
      {SUGGESTED_CONCEPTS.map((concept) => (
        <button
          key={concept}
          onClick={() => onSelect(concept)}
          className="
            px-3.5 py-1.5 text-sm font-medium
            bg-white border border-card-border rounded-full
            text-text-secondary
            hover:border-accent-blue hover:text-accent-blue hover:bg-accent-blue-light
            active:bg-blue-100
            transition-all duration-200 cursor-pointer
          "
        >
          {concept}
        </button>
      ))}
    </div>
  );
}
