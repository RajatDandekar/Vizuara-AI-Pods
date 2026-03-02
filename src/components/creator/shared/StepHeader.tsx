'use client';

interface StepHeaderProps {
  title: string;
  description: string;
  stepNumber: number;
  totalSteps?: number;
}

export default function StepHeader({
  title,
  description,
  stepNumber,
  totalSteps = 8,
}: StepHeaderProps) {
  return (
    <div className="mb-8">
      <div className="flex items-center gap-3 mb-2">
        <span className="text-sm font-medium text-accent-blue bg-accent-blue-light px-3 py-1 rounded-full">
          Step {stepNumber} of {totalSteps}
        </span>
      </div>
      <h1 className="text-3xl font-bold text-foreground">{title}</h1>
      <p className="text-base text-text-secondary mt-1.5">{description}</p>
    </div>
  );
}
