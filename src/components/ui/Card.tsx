'use client';

import { type HTMLAttributes, type ReactNode } from 'react';

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode;
  isActive?: boolean;
  isClickable?: boolean;
}

export default function Card({
  children,
  isActive = false,
  isClickable = false,
  className = '',
  ...props
}: CardProps) {
  return (
    <div
      className={`
        bg-card-bg border rounded-2xl transition-all duration-300 ease-out
        ${isActive
          ? 'border-accent-blue shadow-lg shadow-blue-100 ring-1 ring-accent-blue/20'
          : 'border-card-border shadow-sm hover:shadow-md'
        }
        ${isClickable ? 'cursor-pointer' : ''}
        ${className}
      `}
      {...props}
    >
      {children}
    </div>
  );
}
