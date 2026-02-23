'use client';

import Image from 'next/image';
import type { CuratorInfo } from '@/types/course';

interface Props {
  curator: CuratorInfo;
}

export default function CuratorVideo({ curator }: Props) {
  return (
    <div className="relative rounded-2xl overflow-hidden mb-8">
      {/* Decorative gradient border */}
      <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-accent-blue via-purple-500 to-accent-amber p-[1.5px] -z-10" />

      <div className="bg-card-bg rounded-2xl overflow-hidden border border-card-border">
        {/* Header */}
        <div className="px-6 pt-5 pb-4">
          <div className="flex items-center gap-3 mb-1">
            <div className="flex items-center justify-center w-8 h-8 rounded-full bg-gradient-to-br from-accent-blue to-purple-500">
              <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" />
              </svg>
            </div>
            <div>
              <h2 className="text-sm font-semibold text-foreground tracking-wide">
                Curator of this Module
              </h2>
            </div>
          </div>
        </div>

        {/* Curator info */}
        <div className="px-6 py-5">
          <div className="flex items-start gap-4">
            {/* Avatar — photo with gradient ring, or initials fallback */}
            <div className="flex-shrink-0 w-12 h-12 rounded-full p-[2px] bg-gradient-to-br from-accent-blue to-purple-500">
              {curator.imageUrl ? (
                <Image
                  src={curator.imageUrl}
                  alt={curator.name}
                  width={44}
                  height={44}
                  className="w-full h-full rounded-full object-cover"
                />
              ) : (
                <div className="w-full h-full rounded-full bg-gradient-to-br from-accent-blue to-purple-500 flex items-center justify-center">
                  <span className="text-white font-semibold text-sm">
                    {curator.name.split(' ').map(n => n[0]).join('')}
                  </span>
                </div>
              )}
            </div>
            <div className="flex-1 min-w-0">
              <h3 className="font-semibold text-foreground text-sm">{curator.name}</h3>
              {curator.title && (
                <p className="text-xs text-text-muted mt-0.5">{curator.title}</p>
              )}
              {curator.bio && (
                <p className="text-sm text-text-secondary mt-2 leading-relaxed">{curator.bio}</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
