'use client';

import { useEffect, useState, useRef, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import FadeIn from '@/components/animations/FadeIn';
import Confetti from '@/components/animations/Confetti';
import Breadcrumb from '@/components/navigation/Breadcrumb';
import Button from '@/components/ui/Button';
import CertificateCard from '@/components/certificate/CertificateCard';
import ShareButtons from '@/components/certificate/ShareButtons';
import type { PodCard, CourseCertificateData } from '@/types/course';
import { isCourseComplete, issueCourseCertificate, getCourseCertificate } from '@/lib/progress';
import { useAuth } from '@/context/AuthContext';

interface Props {
  courseSlug: string;
  courseTitle: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  estimatedHours: number;
  livePods: PodCard[];
}

export default function CourseCertificateClient({
  courseSlug,
  courseTitle,
  difficulty,
  estimatedHours,
  livePods,
}: Props) {
  const router = useRouter();
  const { user } = useAuth();
  const [certificate, setCertificate] = useState<CourseCertificateData | null>(null);
  const [showConfetti, setShowConfetti] = useState(false);
  const certificateRef = useRef<HTMLDivElement>(null);

  const totalNotebooks = livePods.reduce((sum, p) => sum + p.notebookCount, 0);

  useEffect(() => {
    if (!user) {
      router.replace('/auth/login');
      return;
    }

    if (!isCourseComplete(courseSlug, livePods)) {
      router.replace(`/courses/${courseSlug}`);
      return;
    }

    const existing = getCourseCertificate(courseSlug);
    const isFirstVisit = !existing;

    const cert = issueCourseCertificate(
      courseSlug,
      courseTitle,
      difficulty,
      estimatedHours,
      livePods.length,
      totalNotebooks,
      user.fullName
    );
    setCertificate(cert);

    if (isFirstVisit) {
      setShowConfetti(true);
      const timer = setTimeout(() => setShowConfetti(false), 5000);
      return () => clearTimeout(timer);
    }
  }, [courseSlug, livePods, courseTitle, difficulty, estimatedHours, totalNotebooks, router, user]);

  const handleDownload = useCallback(async () => {
    if (!certificateRef.current) return;
    const html2canvas = (await import('html2canvas')).default;
    const canvas = await html2canvas(certificateRef.current, {
      scale: 2,
      backgroundColor: null,
      useCORS: true,
    });
    const link = document.createElement('a');
    link.download = `vizuara-course-certificate-${courseSlug}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
  }, [courseSlug]);

  if (!certificate) return null;

  // Adapt CourseCertificateData to CertificateData shape for CertificateCard
  const certForCard = {
    certificateId: certificate.certificateId,
    studentName: certificate.studentName,
    courseTitle: certificate.courseTitle,
    courseSlug: certificate.courseSlug,
    completionDate: certificate.completionDate,
    difficulty: certificate.difficulty,
    estimatedHours: certificate.estimatedHours,
    notebookCount: certificate.totalNotebooks,
  };

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 py-8">
      {showConfetti && <Confetti />}

      <FadeIn>
        <Breadcrumb
          items={[
            { label: 'Courses', href: '/' },
            { label: courseTitle, href: `/courses/${courseSlug}` },
            { label: 'Course Certificate' },
          ]}
        />
      </FadeIn>

      <FadeIn delay={0.1}>
        <div className="text-center mb-8 mt-4">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-accent-amber-light mb-4">
            <svg className="w-8 h-8 text-accent-amber" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 18.75h-9m9 0a3 3 0 013 3h-15a3 3 0 013-3m9 0v-3.375c0-.621-.503-1.125-1.125-1.125h-.871M7.5 18.75v-3.375c0-.621.504-1.125 1.125-1.125h.872m5.007 0H9.497m5.007 0a7.454 7.454 0 01-.982-3.172M9.497 14.25a7.454 7.454 0 00.981-3.172M5.25 4.236c-.982.143-1.954.317-2.916.52A6.003 6.003 0 007.73 9.728M5.25 4.236V4.5c0 2.108.966 3.99 2.48 5.228M5.25 4.236V2.721C7.456 2.41 9.71 2.25 12 2.25c2.291 0 4.545.16 6.75.47v1.516M18.75 4.236c.982.143 1.954.317 2.916.52A6.003 6.003 0 0016.27 9.728M18.75 4.236V4.5c0 2.108-.966 3.99-2.48 5.228m0 0a6.003 6.003 0 01-5.54 0" />
            </svg>
          </div>
          <h1 className="text-2xl sm:text-3xl font-bold text-foreground mb-2">
            Course Complete!
          </h1>
          <p className="text-text-secondary">
            You have completed all {livePods.length} pods in{' '}
            <span className="font-semibold text-foreground">{courseTitle}</span>
          </p>
        </div>
      </FadeIn>

      <FadeIn delay={0.2}>
        <div ref={certificateRef}>
          <CertificateCard certificate={certForCard} />
        </div>
      </FadeIn>

      <FadeIn delay={0.3}>
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mt-8">
          <Button size="lg" onClick={handleDownload}>
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
            </svg>
            Download Certificate
          </Button>
          <ShareButtons certificate={certForCard} certificateRef={certificateRef} />
        </div>
      </FadeIn>

      <FadeIn delay={0.35}>
        <div className="text-center mt-6 mb-8">
          <p className="text-sm text-text-muted">
            Share your achievement with your network and inspire others to learn!
          </p>
        </div>
      </FadeIn>
    </div>
  );
}
