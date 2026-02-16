'use client';

import { useState } from 'react';
import type { CourseCard } from '@/types/course';

type Difficulty = 'all' | 'beginner' | 'intermediate' | 'advanced';

interface DifficultyFilterProps {
  courses: CourseCard[];
  onFilter: (filtered: CourseCard[]) => void;
  dark?: boolean;
}

const filters: { label: string; value: Difficulty }[] = [
  { label: 'All', value: 'all' },
  { label: 'Beginner', value: 'beginner' },
  { label: 'Intermediate', value: 'intermediate' },
  { label: 'Advanced', value: 'advanced' },
];

export default function DifficultyFilter({ courses, onFilter, dark = false }: DifficultyFilterProps) {
  const [active, setActive] = useState<Difficulty>('all');

  function handleClick(value: Difficulty) {
    setActive(value);
    if (value === 'all') {
      onFilter(courses);
    } else {
      onFilter(courses.filter((c) => c.difficulty === value));
    }
  }

  return (
    <div className="flex items-center gap-2 flex-wrap">
      {filters.map((f) => (
        <button
          key={f.value}
          onClick={() => handleClick(f.value)}
          className={`
            px-3.5 py-1.5 rounded-full text-sm font-medium transition-all duration-200
            cursor-pointer
            ${active === f.value
              ? dark
                ? 'bg-blue-500/20 text-blue-300 border border-blue-500/40'
                : 'bg-accent-blue text-white shadow-sm'
              : dark
                ? 'bg-white/5 text-slate-400 border border-white/10 hover:bg-white/10'
                : 'bg-gray-100 text-text-secondary hover:bg-gray-200'
            }
          `}
        >
          {f.label}
        </button>
      ))}
    </div>
  );
}
