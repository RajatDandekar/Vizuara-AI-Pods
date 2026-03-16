'use client';

import { usePathname } from 'next/navigation';
import { useTheme } from '@/context/ThemeContext';

export default function Footer() {
  const pathname = usePathname();
  const { theme } = useTheme();
  const isLandingPage = pathname === '/about' || pathname === '/letter';
  const isDark = isLandingPage || theme === 'dark';

  return (
    <footer className={`border-t mt-auto ${isDark ? 'border-white/5 bg-[#020617]' : 'border-card-border'}`}>
      <div className={`max-w-5xl mx-auto px-4 sm:px-6 py-6 flex items-center justify-between text-xs ${isDark ? 'text-slate-500' : 'text-text-muted'}`}>
        <p>Vizuara — Learn AI from first principles</p>
        <p>Powered by Claude</p>
      </div>
    </footer>
  );
}
