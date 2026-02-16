'use client';

import { type RefObject, useState } from 'react';
import Button from '@/components/ui/Button';
import type { CertificateData } from '@/types/course';

interface ShareButtonsProps {
  certificate: CertificateData;
  certificateRef: RefObject<HTMLDivElement | null>;
}

export default function ShareButtons({ certificate, certificateRef }: ShareButtonsProps) {
  const { courseTitle, courseSlug, notebookCount } = certificate;
  const [hint, setHint] = useState<string | null>(null);

  // --- X (Twitter) ---
  const xText = `I just completed "${courseTitle}" on @VizuaraAI! Learned from an in-depth article, ${notebookCount} hands-on Colab notebooks, and a real-world case study. #AI #MachineLearning`;
  const xUrl = `https://x.com/intent/tweet?${new URLSearchParams({ text: xText }).toString()}`;

  // --- LinkedIn ---
  const linkedInText = `Just wrapped up "${courseTitle}" on Vizuara — went through the full deep-dive article, built ${notebookCount} hands-on notebooks in Colab, and completed a real-world case study from scratch.\n\nIf you're looking to actually understand how these models work (not just the theory), highly recommend checking it out.\n\nhttps://in.linkedin.com/company/vizuara`;
  const linkedInUrl = `https://www.linkedin.com/feed/?shareActive=true&text=${encodeURIComponent(linkedInText)}`;

  /** Capture the certificate card as a PNG File */
  const captureCertificate = async (): Promise<File | null> => {
    if (!certificateRef.current) return null;
    try {
      const html2canvas = (await import('html2canvas')).default;
      const canvas = await html2canvas(certificateRef.current, {
        scale: 2,
        backgroundColor: null,
        useCORS: true,
      });
      const blob = await new Promise<Blob | null>((resolve) =>
        canvas.toBlob(resolve, 'image/png')
      );
      if (!blob) return null;
      return new File([blob], `vizuara-certificate-${courseSlug}.png`, {
        type: 'image/png',
      });
    } catch {
      return null;
    }
  };

  const handleLinkedInShare = async () => {
    const file = await captureCertificate();

    // Try Web Share API (works on mobile + Chrome/Edge desktop)
    // This opens the native OS share sheet with the image pre-attached
    if (file && navigator.canShare?.({ files: [file] })) {
      try {
        await navigator.share({
          files: [file],
          title: `Vizuara Certificate – ${courseTitle}`,
          text: linkedInText,
        });
        return; // User shared or cancelled via native dialog
      } catch {
        // User cancelled or share failed — fall through to fallback
      }
    }

    // Fallback: download the file + open LinkedIn composer
    if (file) {
      const url = URL.createObjectURL(file);
      const link = document.createElement('a');
      link.download = file.name;
      link.href = url;
      link.click();
      URL.revokeObjectURL(url);
    }

    setHint('Certificate saved! Attach it to your LinkedIn post.');
    setTimeout(() => setHint(null), 5000);
    setTimeout(() => {
      window.open(linkedInUrl, '_blank', 'noopener,noreferrer');
    }, 600);
  };

  return (
    <div className="flex items-center gap-3 relative">
      {/* Share on X */}
      <a href={xUrl} target="_blank" rel="noopener noreferrer">
        <Button variant="secondary" size="md">
          <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
            <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
          </svg>
          Share on X
        </Button>
      </a>

      {/* Share on LinkedIn */}
      <div className="relative">
        <Button variant="secondary" size="md" onClick={handleLinkedInShare}>
          <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
            <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
          </svg>
          Share on LinkedIn
        </Button>
        {hint && (
          <div className="absolute -top-12 left-1/2 -translate-x-1/2 bg-foreground text-white text-xs px-3 py-2 rounded-lg whitespace-nowrap shadow-lg animate-fade-in">
            {hint}
          </div>
        )}
      </div>
    </div>
  );
}
