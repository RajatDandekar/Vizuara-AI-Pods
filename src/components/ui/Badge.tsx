'use client';

import { type ReactNode } from 'react';

interface BadgeProps {
  children: ReactNode;
  variant?: 'default' | 'blue' | 'green' | 'amber' | 'red';
  size?: 'sm' | 'md';
}

const variantStyles = {
  default: 'bg-gray-100 text-text-secondary',
  blue: 'bg-accent-blue-light text-accent-blue',
  green: 'bg-accent-green-light text-accent-green',
  amber: 'bg-accent-amber-light text-accent-amber',
  red: 'bg-red-50 text-accent-red',
};

const sizeStyles = {
  sm: 'px-2 py-0.5 text-xs',
  md: 'px-2.5 py-1 text-sm',
};

export default function Badge({
  children,
  variant = 'default',
  size = 'sm',
}: BadgeProps) {
  return (
    <span
      className={`
        inline-flex items-center font-semibold rounded-full
        ${variantStyles[variant]}
        ${sizeStyles[size]}
      `}
    >
      {children}
    </span>
  );
}
