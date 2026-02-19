'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Image from 'next/image';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuth } from '@/context/AuthContext';
import Button from '@/components/ui/Button';
import Badge from '@/components/ui/Badge';
import catalogData from '@/../content/courses/catalog.json';

const ALL_INTEREST_TAGS = Array.from(
  new Set(
    catalogData.courses.flatMap((c: { tags: string[] }) => c.tags).filter(Boolean)
  )
).sort() as string[];

const TAG_LABELS: Record<string, string> = {
  'robotics': 'Robotics',
  'nlp': 'NLP',
  'transformers': 'Transformers',
  'reinforcement-learning': 'Reinforcement Learning',
  'language-models': 'Language Models',
  'computer-vision': 'Computer Vision',
  'distributed-training': 'Distributed Training',
  'foundation-models': 'Foundation Models',
  'diffusion-models': 'Diffusion Models',
  'agents': 'AI Agents',
  'reasoning': 'Reasoning',
  'efficiency': 'Efficiency',
  'graph-networks': 'Graph Networks',
  'alignment': 'AI Alignment',
  'rag': 'RAG',
  'world-models': 'World Models',
  'simulation': 'Simulation',
  'vision-language': 'Vision-Language',
  'systems': 'Systems',
  'gpu': 'GPU Programming',
  'geometric-dl': 'Geometric DL',
};

const EXPERIENCE_LEVELS = [
  { value: 'beginner', label: 'Beginner', desc: 'New to AI/ML, learning the fundamentals' },
  { value: 'intermediate', label: 'Intermediate', desc: 'Comfortable with core concepts, building projects' },
  { value: 'advanced', label: 'Advanced', desc: 'Deep expertise, working on cutting-edge research' },
];

const stepVariants = {
  enter: { opacity: 0, x: 40 },
  center: { opacity: 1, x: 0 },
  exit: { opacity: 0, x: -40 },
};

export default function OnboardingPage() {
  const { user, refreshUser } = useAuth();
  const router = useRouter();

  const [step, setStep] = useState(1);
  const [selectedInterests, setSelectedInterests] = useState<string[]>([]);
  const [experienceLevel, setExperienceLevel] = useState<string>('');
  const [saving, setSaving] = useState(false);

  // Redirect if not logged in or already onboarded
  useEffect(() => {
    if (user === null) {
      router.push('/auth/login');
    }
  }, [user, router]);

  function toggleInterest(tag: string) {
    setSelectedInterests((prev) =>
      prev.includes(tag)
        ? prev.filter((t) => t !== tag)
        : [...prev, tag]
    );
  }

  async function handleComplete() {
    setSaving(true);
    try {
      await fetch('/api/auth/onboarding', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          interests: selectedInterests,
          experienceLevel: experienceLevel || null,
        }),
      });
      await refreshUser();
      router.push('/');
    } catch {
      setSaving(false);
    }
  }

  async function handleSkip() {
    setSaving(true);
    try {
      await fetch('/api/auth/onboarding', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ interests: [], experienceLevel: null }),
      });
      await refreshUser();
      router.push('/');
    } catch {
      setSaving(false);
    }
  }

  // Get recommended courses based on selected interests
  const recommendedCourses = catalogData.courses
    .filter((c: { tags: string[]; status?: string }) =>
      (c.status === 'live' || !c.status) &&
      c.tags.some((t: string) => selectedInterests.includes(t))
    )
    .slice(0, 3);

  if (!user) return null;

  return (
    <div className="min-h-[calc(100vh-3.5rem)] flex items-center justify-center px-4 py-12 bg-gradient-to-br from-slate-50 via-blue-50/30 to-indigo-50/20">
      <div className="w-full max-w-2xl">
        {/* Progress dots */}
        <div className="flex items-center justify-center gap-2 mb-8">
          {[1, 2, 3, 4].map((s) => (
            <div
              key={s}
              className={`h-2 rounded-full transition-all duration-300 ${
                s === step ? 'w-8 bg-accent-blue' : s < step ? 'w-2 bg-accent-blue/50' : 'w-2 bg-gray-300'
              }`}
            />
          ))}
        </div>

        <AnimatePresence mode="wait">
          {/* Step 1: Welcome */}
          {step === 1 && (
            <motion.div
              key="step1"
              variants={stepVariants}
              initial="enter" animate="center" exit="exit"
              transition={{ duration: 0.3, ease: [0.25, 0.1, 0.25, 1] }}
              className="bg-card-bg border border-card-border rounded-2xl p-10 shadow-sm text-center"
            >
              <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-2xl flex items-center justify-center mx-auto mb-6">
                <Image src="/vizuara-logo.png" alt="Vizuara AI Pods" width={48} height={48} className="rounded-lg" />
              </div>
              <h1 className="text-3xl font-bold text-foreground mb-3">
                Welcome, {user.fullName.split(' ')[0]}!
              </h1>
              <p className="text-text-secondary leading-relaxed mb-8 max-w-md mx-auto">
                Let&apos;s personalize your learning experience. We&apos;ll ask you a couple of
                quick questions to recommend the best courses for you.
              </p>
              <div className="flex items-center justify-center gap-3">
                <Button size="lg" onClick={() => setStep(2)}>
                  Get Started
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
                  </svg>
                </Button>
              </div>
              <button
                onClick={handleSkip}
                className="text-sm text-text-muted hover:text-text-secondary mt-4 transition-colors cursor-pointer"
              >
                Skip for now
              </button>
            </motion.div>
          )}

          {/* Step 2: Interest Selection */}
          {step === 2 && (
            <motion.div
              key="step2"
              variants={stepVariants}
              initial="enter" animate="center" exit="exit"
              transition={{ duration: 0.3, ease: [0.25, 0.1, 0.25, 1] }}
              className="bg-card-bg border border-card-border rounded-2xl p-8 shadow-sm"
            >
              <h2 className="text-2xl font-bold text-foreground mb-2 text-center">
                What topics interest you?
              </h2>
              <p className="text-sm text-text-secondary text-center mb-6">
                Pick at least 3 for the best recommendations
              </p>

              <div className="flex flex-wrap gap-2.5 justify-center mb-8">
                {ALL_INTEREST_TAGS.map((tag) => (
                  <button
                    key={tag}
                    onClick={() => toggleInterest(tag)}
                    className={`
                      px-4 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 cursor-pointer
                      ${selectedInterests.includes(tag)
                        ? 'bg-accent-blue text-white shadow-sm ring-2 ring-accent-blue/30 scale-105'
                        : 'bg-gray-100 text-text-secondary hover:bg-gray-200 hover:text-foreground'
                      }
                    `}
                  >
                    {selectedInterests.includes(tag) && (
                      <span className="mr-1.5">&#10003;</span>
                    )}
                    {TAG_LABELS[tag] || tag}
                  </button>
                ))}
              </div>

              {selectedInterests.length > 0 && (
                <p className="text-xs text-text-muted text-center mb-4">
                  {selectedInterests.length} topic{selectedInterests.length !== 1 ? 's' : ''} selected
                </p>
              )}

              <div className="flex items-center justify-between">
                <Button variant="ghost" onClick={() => setStep(1)}>Back</Button>
                <Button onClick={() => setStep(3)} disabled={selectedInterests.length === 0}>
                  Continue
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
                  </svg>
                </Button>
              </div>
            </motion.div>
          )}

          {/* Step 3: Experience Level */}
          {step === 3 && (
            <motion.div
              key="step3"
              variants={stepVariants}
              initial="enter" animate="center" exit="exit"
              transition={{ duration: 0.3, ease: [0.25, 0.1, 0.25, 1] }}
              className="bg-card-bg border border-card-border rounded-2xl p-8 shadow-sm"
            >
              <h2 className="text-2xl font-bold text-foreground mb-2 text-center">
                What&apos;s your experience level?
              </h2>
              <p className="text-sm text-text-secondary text-center mb-8">
                This helps us prioritize the right courses for you
              </p>

              <div className="space-y-3 max-w-md mx-auto mb-8">
                {EXPERIENCE_LEVELS.map((level) => (
                  <button
                    key={level.value}
                    onClick={() => setExperienceLevel(level.value)}
                    className={`
                      w-full text-left px-5 py-4 rounded-xl border-2 transition-all duration-200 cursor-pointer
                      ${experienceLevel === level.value
                        ? 'border-accent-blue bg-accent-blue-light/50 shadow-sm'
                        : 'border-card-border hover:border-gray-300 hover:bg-gray-50'
                      }
                    `}
                  >
                    <p className={`font-semibold text-sm ${experienceLevel === level.value ? 'text-accent-blue' : 'text-foreground'}`}>
                      {level.label}
                    </p>
                    <p className="text-xs text-text-muted mt-0.5">{level.desc}</p>
                  </button>
                ))}
              </div>

              <div className="flex items-center justify-between">
                <Button variant="ghost" onClick={() => setStep(2)}>Back</Button>
                <div className="flex items-center gap-3">
                  <button
                    onClick={() => setStep(4)}
                    className="text-sm text-text-muted hover:text-text-secondary transition-colors cursor-pointer"
                  >
                    Skip
                  </button>
                  <Button onClick={() => setStep(4)}>
                    Continue
                  </Button>
                </div>
              </div>
            </motion.div>
          )}

          {/* Step 4: Confirmation */}
          {step === 4 && (
            <motion.div
              key="step4"
              variants={stepVariants}
              initial="enter" animate="center" exit="exit"
              transition={{ duration: 0.3, ease: [0.25, 0.1, 0.25, 1] }}
              className="bg-card-bg border border-card-border rounded-2xl p-8 shadow-sm text-center"
            >
              <div className="w-16 h-16 bg-accent-green-light rounded-2xl flex items-center justify-center mx-auto mb-6">
                <svg className="w-8 h-8 text-accent-green" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>

              <h2 className="text-2xl font-bold text-foreground mb-2">
                You&apos;re all set!
              </h2>
              <p className="text-sm text-text-secondary mb-6">
                We&apos;ve personalized your feed based on your preferences.
              </p>

              {/* Selected interests */}
              {selectedInterests.length > 0 && (
                <div className="mb-6">
                  <p className="text-xs text-text-muted mb-2">Your interests</p>
                  <div className="flex flex-wrap gap-2 justify-center">
                    {selectedInterests.map((tag) => (
                      <Badge key={tag} variant="blue" size="sm">
                        {TAG_LABELS[tag] || tag}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              {/* Recommended courses preview */}
              {recommendedCourses.length > 0 && (
                <div className="mb-8">
                  <p className="text-xs text-text-muted mb-3">Top picks for you</p>
                  <div className="space-y-2">
                    {recommendedCourses.map((course: { slug: string; title: string; description: string }) => (
                      <div
                        key={course.slug}
                        className="bg-gray-50 rounded-xl px-4 py-3 text-left"
                      >
                        <p className="text-sm font-medium text-foreground">{course.title}</p>
                        <p className="text-xs text-text-muted line-clamp-1">{course.description}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <Button size="lg" onClick={handleComplete} isLoading={saving}>
                Explore Courses
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
                </svg>
              </Button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
