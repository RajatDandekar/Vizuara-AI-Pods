'use client';

import Link from 'next/link';
import Image from 'next/image';
import { usePathname } from 'next/navigation';
import { useAuth } from '@/context/AuthContext';
import UserDropdown from '@/components/auth/UserDropdown';

const VIZUARA_URL = process.env.NEXT_PUBLIC_VIZUARA_URL || 'https://vizuara.ai';
const PODS_CALLBACK_URL = process.env.NEXT_PUBLIC_PODS_CALLBACK_URL || 'https://pods.vizuara.ai/api/auth/session';
const darkPages = ['/about', '/letter'];

export default function Header() {
  const pathname = usePathname();
  const { user, loading } = useAuth();
  const isDark = darkPages.includes(pathname);

  return (
    <header
      className={`sticky top-0 z-50 backdrop-blur-md border-b transition-colors ${
        isDark
          ? 'bg-slate-900/80 border-white/10'
          : 'bg-white/80 border-card-border'
      }`}
    >
      <div className="max-w-6xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
        <Link
          href="/"
          className={`flex items-center gap-2.5 hover:opacity-80 transition-opacity ${
            isDark ? 'text-white' : 'text-foreground'
          }`}
        >
          <Image
            src="/vizuara-logo.png"
            alt="Vizuara"
            width={28}
            height={28}
            className="rounded-md"
          />
          <span className="font-bold text-lg tracking-tight">Vizuara AI Pods</span>
        </Link>

        <nav className="flex items-center gap-6">
          <Link
            href="/"
            className={`text-sm transition-colors ${
              pathname === '/'
                ? isDark ? 'text-white font-medium' : 'text-foreground font-medium'
                : isDark ? 'text-blue-200/60 hover:text-white' : 'text-text-muted hover:text-foreground'
            }`}
          >
            Courses
          </Link>
          <Link
            href="/discover"
            className={`text-sm transition-colors ${
              pathname.startsWith('/discover')
                ? isDark ? 'text-white font-medium' : 'text-foreground font-medium'
                : isDark ? 'text-blue-200/60 hover:text-white' : 'text-text-muted hover:text-foreground'
            }`}
          >
            Discover
          </Link>
          <Link
            href="/about"
            className={`text-sm transition-colors ${
              pathname === '/about'
                ? isDark ? 'text-white font-medium' : 'text-foreground font-medium'
                : isDark ? 'text-blue-200/60 hover:text-white' : 'text-text-muted hover:text-foreground'
            }`}
          >
            About
          </Link>
          <Link
            href="/pricing"
            className={`text-sm transition-colors ${
              pathname === '/pricing'
                ? isDark ? 'text-white font-medium' : 'text-foreground font-medium'
                : isDark ? 'text-blue-200/60 hover:text-white' : 'text-text-muted hover:text-foreground'
            }`}
          >
            Pricing
          </Link>

          {/* Auth section */}
          {!loading && (
            <>
              {user ? (
                <UserDropdown isHome={isDark} />
              ) : (
                <div className="flex items-center gap-3">
                  <a
                    href={`${VIZUARA_URL}/auth/login?redirect=${encodeURIComponent(PODS_CALLBACK_URL)}`}
                    className={`text-sm font-medium transition-colors ${
                      isDark
                        ? 'text-blue-200/80 hover:text-white'
                        : 'text-text-secondary hover:text-foreground'
                    }`}
                  >
                    Log in
                  </a>
                  <a
                    href={`${VIZUARA_URL}/auth/signup?redirect=${encodeURIComponent(PODS_CALLBACK_URL)}`}
                    className="text-sm font-medium bg-accent-blue text-white px-4 py-1.5 rounded-lg hover:bg-blue-700 transition-colors"
                  >
                    Sign Up
                  </a>
                </div>
              )}
            </>
          )}
        </nav>
      </div>
    </header>
  );
}
