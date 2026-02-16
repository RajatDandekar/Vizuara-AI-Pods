'use client';

import { useState, type FormEvent } from 'react';
import { useRouter } from 'next/navigation';
import Input from '@/components/ui/Input';
import Button from '@/components/ui/Button';
import ConceptSuggestions from './ConceptSuggestions';

interface ConceptInputProps {
  basePath?: string;
}

export default function ConceptInput({ basePath = '/discover' }: ConceptInputProps) {
  const [concept, setConcept] = useState('');
  const router = useRouter();

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const trimmed = concept.trim();
    if (!trimmed) return;
    router.push(`${basePath}?concept=${encodeURIComponent(trimmed)}`);
  }

  function handleSuggestionSelect(suggestion: string) {
    setConcept(suggestion);
    router.push(`${basePath}?concept=${encodeURIComponent(suggestion)}`);
  }

  return (
    <div className="w-full max-w-xl mx-auto space-y-6">
      <form onSubmit={handleSubmit} className="flex gap-3">
        <Input
          value={concept}
          onChange={(e) => setConcept(e.target.value)}
          placeholder="e.g., Transformers, GANs, Diffusion Models..."
          autoFocus
        />
        <Button type="submit" size="lg" disabled={!concept.trim()}>
          Explore
        </Button>
      </form>

      <div className="text-center">
        <p className="text-sm text-text-muted mb-3">Or try a popular concept:</p>
        <ConceptSuggestions onSelect={handleSuggestionSelect} />
      </div>
    </div>
  );
}
