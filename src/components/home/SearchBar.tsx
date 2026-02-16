'use client';

interface SearchBarProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  dark?: boolean;
}

export default function SearchBar({ value, onChange, placeholder = 'Search courses...', dark = false }: SearchBarProps) {
  return (
    <div className="relative">
      <svg
        className={`absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 ${dark ? 'text-slate-500' : 'text-text-muted'}`}
        fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
      >
        <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z" />
      </svg>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className={`w-full pl-10 pr-4 py-2.5 text-sm rounded-xl focus:outline-none focus:ring-2 transition-all duration-200 ${
          dark
            ? 'text-white bg-white/5 border border-white/10 placeholder:text-slate-500 focus:ring-blue-500/30 focus:border-blue-500/50'
            : 'text-foreground bg-white border border-card-border placeholder:text-text-muted focus:ring-accent-blue/30 focus:border-accent-blue'
        }`}
      />
      {value && (
        <button
          onClick={() => onChange('')}
          className={`absolute right-3 top-1/2 -translate-y-1/2 transition-colors cursor-pointer ${
            dark ? 'text-slate-500 hover:text-white' : 'text-text-muted hover:text-foreground'
          }`}
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      )}
    </div>
  );
}
