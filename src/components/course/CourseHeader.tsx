'use client';

import Badge from '@/components/ui/Badge';

interface CourseHeaderProps {
  title: string;
  description: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  estimatedHours: number;
  notebookCount: number;
  tags: string[];
}

const difficultyVariant: Record<string, 'green' | 'blue' | 'amber'> = {
  beginner: 'green',
  intermediate: 'blue',
  advanced: 'amber',
};

export default function CourseHeader({
  title,
  description,
  difficulty,
  estimatedHours,
  notebookCount,
  tags,
}: CourseHeaderProps) {
  return (
    <div className="mb-8">
      <h1 className="text-3xl sm:text-4xl font-bold text-foreground tracking-tight mb-4">
        {title}
      </h1>
      <p className="text-text-secondary leading-relaxed mb-5 max-w-3xl">
        {description}
      </p>
      <div className="flex items-center gap-3 flex-wrap">
        <Badge variant={difficultyVariant[difficulty]} size="md">
          {difficulty}
        </Badge>
        <span className="text-sm text-text-muted flex items-center gap-1.5">
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          ~{estimatedHours} hours
        </span>
        {notebookCount > 0 && (
          <span className="text-sm text-text-muted flex items-center gap-1.5">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
            </svg>
            {notebookCount} notebooks
          </span>
        )}
        {tags.map((tag) => (
          <span key={tag} className="text-xs px-2.5 py-1 rounded-full bg-gray-100 text-text-secondary">
            {tag}
          </span>
        ))}
      </div>
    </div>
  );
}
