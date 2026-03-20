'use client';

import Image from 'next/image';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { useAuth } from '@/context/AuthContext';
import { useSubscription } from '@/context/SubscriptionContext';
import FadeIn from '@/components/animations/FadeIn';

const VIZUARA_URL = process.env.NEXT_PUBLIC_VIZUARA_URL || 'https://vizuara.ai';
const PODS_CALLBACK_URL = process.env.NEXT_PUBLIC_PODS_CALLBACK_URL || 'https://pods.vizuara.ai/api/auth/session';

const stats = [
  { label: 'Chapters', value: '13' },
  { label: 'Topics', value: '73' },
  { label: 'Diagrams', value: '200+' },
  { label: 'Pages', value: '400+' },
];

const benefits = [
  'Visual-first learning — every concept explained with diagrams',
  'From neurons to production — build understanding from scratch',
  'Covers Transformers, Diffusion, RL, Agents, and more',
  'Perfect companion to hands-on courses',
];

const chapter1Topics = [
  'Perceptrons', 'Activation Functions', 'Backpropagation',
  'Loss Functions', 'Gradient Descent', 'Neural Network from Scratch',
];

export default function BookSection() {
  const { user } = useAuth();
  const { enrolled, loading } = useSubscription();

  return (
    <FadeIn delay={0.1}>
      <div className="relative overflow-hidden rounded-3xl border border-indigo-200/60 bg-gradient-to-br from-indigo-50/80 via-white to-violet-50/80 p-8 sm:p-10">
        {/* Header badge */}
        <div className="flex items-center gap-2 mb-6">
          <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-indigo-100 text-indigo-700 text-xs font-medium">
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" />
            </svg>
            Free with subscription
          </div>
        </div>

        {/* Two-column layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12 items-center">
          {/* Left — 3D Book Cover */}
          <div className="flex justify-center">
            <div
              className="relative"
              style={{ perspective: '1200px' }}
            >
              <motion.div
                className="relative"
                style={{
                  transformStyle: 'preserve-3d',
                  transform: 'rotateY(-12deg)',
                }}
                whileHover={{ transform: 'rotateY(-4deg)' }}
                transition={{ duration: 0.4, ease: [0.25, 0.1, 0.25, 1] as [number, number, number, number] }}
              >
                {/* Book spine */}
                <div
                  className="absolute top-0 left-0 h-full w-4 bg-gradient-to-b from-indigo-700 to-violet-800 rounded-l-sm"
                  style={{
                    transform: 'rotateY(90deg) translateZ(8px) translateX(-8px)',
                    transformOrigin: 'left center',
                  }}
                />
                {/* Shadow */}
                <div
                  className="absolute -bottom-4 left-4 right-0 h-6 rounded-xl"
                  style={{
                    background: 'radial-gradient(ellipse at center, rgba(99,102,241,0.2) 0%, transparent 70%)',
                    filter: 'blur(8px)',
                  }}
                />
                {/* Cover image */}
                <div className="relative w-[240px] sm:w-[280px] aspect-[3/4] rounded-lg overflow-hidden shadow-xl border border-indigo-200/40">
                  <Image
                    src="/book/cover.png"
                    alt="Vizuara: A Complete Visual Guide to Mastering Modern AI and LLMs"
                    fill
                    className="object-cover"
                    sizes="280px"
                    priority
                  />
                </div>
              </motion.div>
            </div>
          </div>

          {/* Right — Info */}
          <div>
            <h2 className="text-2xl sm:text-3xl font-bold text-slate-900 mb-2">
              Vizuara: A Complete Visual Guide to AI
            </h2>
            <p className="text-base text-slate-500 mb-5">
              Master modern AI and LLMs through 200+ hand-crafted diagrams — from neural networks to production systems.
            </p>

            {/* Stats row */}
            <div className="flex flex-wrap gap-4 mb-5">
              {stats.map((s) => (
                <div key={s.label} className="text-center">
                  <div className="text-xl font-bold text-indigo-700">{s.value}</div>
                  <div className="text-xs text-slate-400">{s.label}</div>
                </div>
              ))}
            </div>

            {/* Benefits */}
            <ul className="space-y-2 mb-6">
              {benefits.map((b) => (
                <li key={b} className="flex items-start gap-2 text-sm text-slate-600">
                  <svg className="w-4 h-4 text-indigo-500 mt-0.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
                  </svg>
                  {b}
                </li>
              ))}
            </ul>

            {/* CTA */}
            <BookCTA user={user} enrolled={enrolled} loading={loading} />
          </div>
        </div>

        {/* Sample Chapter Preview */}
        <div className="mt-10 pt-8 border-t border-indigo-100">
          <h3 className="text-lg font-semibold text-slate-900 mb-1">
            Peek Inside — Chapter 1: Build a Neural Network from Scratch
          </h3>
          <p className="text-sm text-slate-500 mb-5">
            See how every concept is explained visually, step by step.
          </p>

          {/* Fanned sample pages */}
          <div className="flex justify-center mb-6">
            <div className="relative h-[200px] sm:h-[260px] w-full max-w-md">
              {[1, 2, 3].map((n, i) => (
                <motion.div
                  key={n}
                  className="absolute top-0 left-1/2 w-[160px] sm:w-[200px] aspect-[3/4] rounded-lg overflow-hidden shadow-lg border border-slate-200 bg-white"
                  style={{
                    transform: `translateX(-50%) rotate(${(i - 1) * 6}deg) translateX(${(i - 1) * 40}px)`,
                    zIndex: 3 - i,
                  }}
                  whileHover={{ y: -8, scale: 1.02 }}
                  transition={{ duration: 0.2 }}
                >
                  <Image
                    src={`/book/sample-${n}.png`}
                    alt={`Sample page ${n} from Chapter 1`}
                    fill
                    className="object-cover"
                    sizes="200px"
                  />
                </motion.div>
              ))}
            </div>
          </div>

          {/* Chapter 1 topics */}
          <div className="flex flex-wrap justify-center gap-2 mb-4">
            {chapter1Topics.map((t) => (
              <span key={t} className="px-3 py-1 rounded-full text-xs font-medium bg-indigo-50 text-indigo-600 border border-indigo-100">
                {t}
              </span>
            ))}
          </div>

          <div className="text-center">
            <Link
              href="/book"
              className="text-sm font-medium text-indigo-600 hover:text-indigo-800 transition-colors"
            >
              See all 13 chapters →
            </Link>
          </div>
        </div>
      </div>
    </FadeIn>
  );
}

function BookCTA({
  user,
  enrolled,
  loading,
}: {
  user: { id: string } | null;
  enrolled: boolean;
  loading: boolean;
}) {
  if (loading) {
    return (
      <div className="flex justify-start">
        <div className="w-6 h-6 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  if (enrolled) {
    return (
      <a
        href="/api/book/download"
        className="inline-flex items-center gap-2 px-6 py-3 bg-indigo-600 text-white font-medium rounded-xl hover:bg-indigo-700 transition-colors shadow-sm"
      >
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
        </svg>
        Download Free PDF
      </a>
    );
  }

  if (user) {
    return (
      <a
        href={`${VIZUARA_URL}/courses/ai-pods`}
        className="inline-flex items-center gap-2 px-6 py-3 bg-indigo-600 text-white font-medium rounded-xl hover:bg-indigo-700 transition-colors shadow-sm"
      >
        Subscribe to Access
      </a>
    );
  }

  return (
    <a
      href={`${VIZUARA_URL}/auth/signup?redirect=${encodeURIComponent(PODS_CALLBACK_URL)}`}
      className="inline-flex items-center gap-2 px-6 py-3 bg-indigo-600 text-white font-medium rounded-xl hover:bg-indigo-700 transition-colors shadow-sm"
    >
      Sign Up to Access
    </a>
  );
}
