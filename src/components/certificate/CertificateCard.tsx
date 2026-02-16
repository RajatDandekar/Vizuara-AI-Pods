'use client';

import Badge from '@/components/ui/Badge';
import type { CertificateData } from '@/types/course';

interface CertificateCardProps {
  certificate: CertificateData;
}

export default function CertificateCard({ certificate }: CertificateCardProps) {
  const formattedDate = new Date(certificate.completionDate).toLocaleDateString(
    'en-US',
    { year: 'numeric', month: 'long', day: 'numeric' }
  );

  const difficultyVariant: Record<string, 'green' | 'blue' | 'amber'> = {
    beginner: 'green',
    intermediate: 'blue',
    advanced: 'amber',
  };

  return (
    <div className="w-full max-w-[800px] mx-auto">
      {/* Outer gold gradient border */}
      <div className="p-[3px] bg-gradient-to-br from-amber-400 via-yellow-300 to-amber-500 rounded-2xl shadow-lg">
        <div className="bg-white rounded-[13px] relative" style={{ aspectRatio: '16/9' }}>
          {/* Inner decorative frame */}
          <div className="absolute inset-4 border border-amber-200/60 rounded-xl pointer-events-none" />

          {/* Content — flex column to distribute vertically */}
          <div className="absolute inset-0 flex flex-col items-center justify-between p-6 sm:p-8">
            {/* Top: Logo + title */}
            <div className="text-center pt-1">
              {/* Use <img> for html2canvas compatibility */}
              <img
                src="/vizuara-logo.png"
                alt="Vizuara"
                width={48}
                height={48}
                className="mx-auto mb-2 rounded-lg"
              />
              <h2 className="text-xs sm:text-sm font-bold text-accent-amber uppercase tracking-[0.2em]">
                Certificate of Completion
              </h2>
            </div>

            {/* Center: Student + Course */}
            <div className="text-center flex-1 flex flex-col items-center justify-center py-1">
              <p className="text-[11px] sm:text-xs text-text-muted mb-1">This certifies that</p>
              <h3 className="text-xl sm:text-2xl font-bold text-foreground mb-1">
                {certificate.studentName}
              </h3>
              <p className="text-[11px] sm:text-xs text-text-muted mb-2 sm:mb-3">
                has successfully completed
              </p>
              <h4 className="text-base sm:text-xl font-semibold text-accent-blue mb-2 sm:mb-3 max-w-md leading-tight px-4">
                {certificate.courseTitle}
              </h4>
              <div className="flex items-center gap-2 flex-wrap justify-center">
                <Badge variant={difficultyVariant[certificate.difficulty]} size="sm">
                  {certificate.difficulty}
                </Badge>
                <span className="text-[10px] sm:text-xs text-text-muted">
                  {certificate.estimatedHours} hours
                </span>
                {certificate.notebookCount > 0 && (
                  <span className="text-[10px] sm:text-xs text-text-muted">
                    {certificate.notebookCount} notebooks
                  </span>
                )}
              </div>
            </div>

            {/* Bottom: Date + ID row, then 3 signatures */}
            <div className="w-full pb-1">
              <div className="flex items-center justify-between mb-2 sm:mb-3 px-1">
                <div className="text-left">
                  <p className="text-[9px] sm:text-[10px] text-text-muted">Issued {formattedDate}</p>
                </div>
                <div className="text-right">
                  <p className="text-[9px] sm:text-[10px] font-mono text-text-muted">
                    {certificate.certificateId}
                  </p>
                </div>
              </div>
              <div className="flex items-end justify-center gap-6 sm:gap-10">
                {[
                  { name: 'Dr. Raj Dandekar', sig: 'Raj Dandekar', title: 'Co-founder' },
                  { name: 'Dr. Rajat Dandekar', sig: 'Rajat Dandekar', title: 'Co-founder' },
                  { name: 'Dr. Sreedath Panat', sig: 'Sreedath Panat', title: 'Co-founder' },
                ].map((signer) => (
                  <div key={signer.name} className="text-center">
                    <p
                      className="text-base sm:text-lg text-foreground/80 mb-0"
                      style={{ fontFamily: "'Dancing Script', cursive", fontWeight: 700 }}
                    >
                      {signer.sig}
                    </p>
                    <div className="w-20 sm:w-28 border-b border-foreground/30 mb-1 mx-auto" />
                    <p className="text-[9px] sm:text-[11px] font-semibold text-foreground">
                      {signer.name}
                    </p>
                    <p className="text-[7px] sm:text-[9px] text-text-muted">
                      {signer.title}, Vizuara Technologies
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
