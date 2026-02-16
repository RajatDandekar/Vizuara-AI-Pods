'use client';

import GradientMeshBg from '@/components/home/GradientMeshBg';
import FoundersLetter from '@/components/home/FoundersLetter';

export default function LetterPage() {
  return (
    <div className="dark-landing min-h-screen" style={{ background: '#020617' }}>
      <GradientMeshBg variant="dark" />
      <div className="relative max-w-4xl mx-auto px-4 sm:px-6 pt-16 pb-24" style={{ zIndex: 1 }}>
        <FoundersLetter />
      </div>
    </div>
  );
}
