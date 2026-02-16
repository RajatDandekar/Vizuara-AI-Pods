'use client';

import type { SectionStatus } from '@/types/paper';

interface Step {
  key: string;
  label: string;
  status: SectionStatus;
  generated: boolean;
}

interface SectionStepperProps {
  steps: Step[];
  activeKey: string;
  onStepClick: (key: string) => void;
}

export default function SectionStepper({ steps, activeKey, onStepClick }: SectionStepperProps) {
  return (
    <div className="flex items-center justify-between mb-8">
      {steps.map((step, i) => {
        const isActive = activeKey === step.key;
        const isStreaming = step.status === 'streaming' || step.status === 'loading';
        const isDone = step.status === 'complete';
        const isError = step.status === 'error';

        return (
          <div key={step.key} className="flex items-center flex-1">
            {/* Step circle + label — clickable */}
            <button
              type="button"
              onClick={() => onStepClick(step.key)}
              className="flex flex-col items-center gap-1.5 cursor-pointer group"
            >
              <div
                className={`
                  w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold transition-all duration-300
                  ${isDone
                    ? isActive
                      ? 'bg-accent-green text-white ring-4 ring-accent-green/20'
                      : 'bg-accent-green text-white'
                    : isStreaming
                      ? 'bg-accent-blue text-white ring-4 ring-accent-blue/20'
                      : isError
                        ? isActive
                          ? 'bg-accent-red text-white ring-4 ring-accent-red/20'
                          : 'bg-accent-red text-white'
                        : isActive
                          ? 'bg-accent-blue text-white ring-4 ring-accent-blue/20'
                          : step.generated
                            ? 'bg-accent-blue/10 text-accent-blue group-hover:bg-accent-blue/20'
                            : 'bg-gray-100 text-text-muted group-hover:bg-gray-200'
                  }
                `}
              >
                {isDone ? (
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                  </svg>
                ) : isStreaming ? (
                  <svg className="w-4.5 h-4.5 animate-spin" viewBox="0 0 24 24" fill="none">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                ) : (
                  i + 1
                )}
              </div>
              <span
                className={`text-xs font-medium whitespace-nowrap transition-colors ${
                  isActive ? 'text-accent-blue' :
                  isDone ? 'text-accent-green' :
                  isStreaming ? 'text-accent-blue' :
                  'text-text-muted group-hover:text-foreground'
                }`}
              >
                {step.label}
              </span>
            </button>

            {/* Connector line */}
            {i < steps.length - 1 && (
              <div className="flex-1 mx-3 h-0.5 rounded-full bg-gray-200 relative overflow-hidden mt-[-20px]">
                <div
                  className={`absolute inset-y-0 left-0 rounded-full transition-all duration-700 ${
                    isDone ? 'w-full bg-accent-green' : isStreaming ? 'w-1/2 bg-accent-blue' : 'w-0'
                  }`}
                />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
