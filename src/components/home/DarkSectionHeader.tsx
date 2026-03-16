'use client';

interface DarkSectionHeaderProps {
  title: string;
  subtitle?: string;
  className?: string;
}

export default function DarkSectionHeader({ title, subtitle, className = '' }: DarkSectionHeaderProps) {
  return (
    <div className={`mb-5 ${className}`}>
      <h2 className="text-xl font-bold text-foreground">{title}</h2>
      {subtitle && <p className="text-sm text-text-muted mt-0.5">{subtitle}</p>}
    </div>
  );
}
