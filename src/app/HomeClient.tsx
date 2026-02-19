'use client';

import { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { AnimatePresence } from 'framer-motion';
import FadeIn from '@/components/animations/FadeIn';
import DifficultyFilter from '@/components/catalog/DifficultyFilter';
import GradientMeshBg from '@/components/home/GradientMeshBg';
import TextMorph from '@/components/home/TextMorph';
import AnimatedCounter from '@/components/home/AnimatedCounter';
import CourseCarousel from '@/components/home/CourseCarousel';
import DarkSectionHeader from '@/components/home/DarkSectionHeader';
import FoundersLetter from '@/components/home/FoundersLetter';
import SearchBar from '@/components/home/SearchBar';
import PodExpandedCard from '@/components/home/PodExpandedCard';
import { useAuth } from '@/context/AuthContext';
import type { CourseCard, PodCard } from '@/types/course';
import { getCourseCompletion } from '@/lib/progress';

// Showcase courses — real examples shown to visitors (optimized images in /showcase)
const showcaseCourses = [
  {
    title: '5D Parallelism from Scratch',
    hero: '/showcase/5dp-hero.png',
    article: 'How modern LLMs are trained across thousands of GPUs — understanding Data, Tensor, Pipeline, Sequence, and Expert Parallelism from first principles.',
    notebooks: [
      'Why Do We Need Parallelism?',
      'Data Parallelism — "Hire More Chefs"',
      'Tensor Parallelism — "Split the Recipe"',
      'Pipeline Parallelism — "The Assembly Line"',
      'Sequence & Expert Parallelism',
      'The 5D Parallelism Grid',
    ],
    caseStudy: 'Optimizing 5D Parallelism for Training a Financial Domain LLM',
  },
  {
    title: 'Diffusion LLMs from Scratch',
    hero: '/showcase/dllm-hero.png',
    article: 'What if language models could generate all tokens at once — like image diffusion, but for text?',
    notebooks: [
      'Image Diffusion Foundations',
      'Masked Diffusion for Text',
      'Training a Diffusion LLM',
      'Generation: Iterative Unmasking',
    ],
    caseStudy: 'Real-Time Code Completion with Diffusion Language Models',
  },
  {
    title: 'VLAs for Autonomous Driving',
    hero: '/showcase/vlad-hero.png',
    article: 'How combining vision, language understanding, and action generation is reshaping autonomous driving — explained from scratch.',
    notebooks: [
      'Vision Encoders & Patch Embeddings',
      'Action Tokenization & Behavioral Cloning',
      'Building a Mini VLA',
      'Diffusion Action Decoder',
    ],
    caseStudy: 'Vision-Language-Action Models for Autonomous Port Terminal Tractors',
  },
];

// Topic pills shown to guests as a sneak peek
const topicPills = [
  'Diffusion Models', 'Reinforcement Learning', 'World Models', 'Robotics',
  'Transformers', 'RAG', 'Reasoning', 'AI Agents', 'NLP', 'Computer Vision',
];

const difficultyColor: Record<string, string> = {
  beginner: 'bg-emerald-50 text-emerald-700 border-emerald-200',
  intermediate: 'bg-blue-50 text-blue-700 border-blue-200',
  advanced: 'bg-amber-50 text-amber-700 border-amber-200',
};

const HOVER_OPEN_DELAY = 400;
const HOVER_CLOSE_DELAY = 300;

interface HomeClientProps {
  courses: CourseCard[];
  coursePods: Record<string, PodCard[]>;
  courseAllPods: Record<string, PodCard[]>;
  draftCourses: CourseCard[];
}

export default function HomeClient({
  courses: allCourses,
  coursePods,
  courseAllPods,
  draftCourses,
}: HomeClientProps) {
  const { user, loading } = useAuth();

  const [completions, setCompletions] = useState<Record<string, number>>({});
  const [search, setSearch] = useState('');
  const [browseFiltered, setBrowseFiltered] = useState<CourseCard[]>([]);

  // --- Hover expansion state for pods ---
  const [expandedPod, setExpandedPod] = useState<{ courseSlug: string; courseTitle: string; pod: PodCard } | null>(null);
  const openTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const closeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (openTimerRef.current) clearTimeout(openTimerRef.current);
      if (closeTimerRef.current) clearTimeout(closeTimerRef.current);
    };
  }, []);

  const handlePodMouseEnter = useCallback((courseSlug: string, courseTitle: string, pod: PodCard) => {
    if (closeTimerRef.current) {
      clearTimeout(closeTimerRef.current);
      closeTimerRef.current = null;
    }
    if (expandedPod?.pod.slug === pod.slug && expandedPod?.courseSlug === courseSlug) return;
    if (openTimerRef.current) clearTimeout(openTimerRef.current);
    openTimerRef.current = setTimeout(() => {
      setExpandedPod({ courseSlug, courseTitle, pod });
      openTimerRef.current = null;
    }, HOVER_OPEN_DELAY);
  }, [expandedPod]);

  const handlePodMouseLeave = useCallback(() => {
    if (openTimerRef.current) {
      clearTimeout(openTimerRef.current);
      openTimerRef.current = null;
    }
    if (expandedPod) {
      closeTimerRef.current = setTimeout(() => {
        setExpandedPod(null);
        closeTimerRef.current = null;
      }, HOVER_CLOSE_DELAY);
    }
  }, [expandedPod]);

  const handleOverlayMouseEnter = useCallback(() => {
    if (closeTimerRef.current) {
      clearTimeout(closeTimerRef.current);
      closeTimerRef.current = null;
    }
  }, []);

  const handleOverlayMouseLeave = useCallback(() => {
    closeTimerRef.current = setTimeout(() => {
      setExpandedPod(null);
      closeTimerRef.current = null;
    }, HOVER_CLOSE_DELAY);
  }, []);

  const handleOverlayClose = useCallback(() => {
    if (openTimerRef.current) clearTimeout(openTimerRef.current);
    if (closeTimerRef.current) clearTimeout(closeTimerRef.current);
    setExpandedPod(null);
  }, []);

  const liveCourses = useMemo(
    () => allCourses.filter((c) => c.status === 'live' || !c.status),
    [allCourses]
  );

  useEffect(() => {
    if (!user) return;
    const comps: Record<string, number> = {};
    for (const course of liveCourses) {
      const pods = coursePods[course.slug] || [];
      if (pods.length > 0) {
        const result = getCourseCompletion(course.slug, pods);
        if (result.percentage > 0) {
          comps[course.slug] = result.percentage;
        }
      }
    }
    setCompletions(comps);
    setBrowseFiltered(liveCourses);
  }, [liveCourses, user, coursePods]);

  const recommendedCourses = useMemo(() => {
    if (!user || user.interests.length === 0) return [];
    return [...liveCourses]
      .map((course) => {
        const score = course.tags.filter((t) => user.interests.includes(t)).length;
        return { course, score };
      })
      .filter(({ score }) => score > 0)
      .sort((a, b) => b.score - a.score)
      .map(({ course }) => course);
  }, [liveCourses, user]);

  const continueLearning = useMemo(() => {
    return liveCourses.filter((c) => {
      const comp = completions[c.slug];
      return comp && comp > 0 && comp < 100;
    });
  }, [liveCourses, completions]);

  // Coming soon pods: pods in allPods but not in livePods
  const comingSoonPods = useMemo(() => {
    const result: { courseSlug: string; courseTitle: string; pod: PodCard }[] = [];
    for (const course of liveCourses) {
      const live = coursePods[course.slug] || [];
      const all = courseAllPods[course.slug] || [];
      const liveSlugs = new Set(live.map((p) => p.slug));
      for (const pod of all) {
        if (!liveSlugs.has(pod.slug)) {
          result.push({ courseSlug: course.slug, courseTitle: course.title, pod });
        }
      }
    }
    // Also include pods from draft courses
    for (const course of draftCourses) {
      const all = courseAllPods[course.slug] || [];
      for (const pod of all) {
        result.push({ courseSlug: course.slug, courseTitle: course.title, pod });
      }
    }
    return result;
  }, [liveCourses, draftCourses, coursePods, courseAllPods]);

  const searchedCourses = useMemo(() => {
    if (!search.trim()) return browseFiltered;
    const q = search.toLowerCase();
    return browseFiltered.filter(
      (c) =>
        c.title.toLowerCase().includes(q) ||
        c.description.toLowerCase().includes(q) ||
        c.tags.some((t) => t.toLowerCase().includes(q))
    );
  }, [browseFiltered, search]);

  const totalNotebooks = liveCourses.reduce((sum, c) => sum + (c.totalNotebooks || c.notebookCount || 0), 0);

  // Don't render until auth state is resolved to avoid flash
  if (loading) {
    return (
      <div className="light-landing min-h-screen flex items-center justify-center" style={{ background: '#fafafa' }}>
        <GradientMeshBg />
        <div className="relative" style={{ zIndex: 1 }}>
          <div className="w-8 h-8 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
        </div>
      </div>
    );
  }

  return (
    <div className="light-landing min-h-screen" style={{ background: '#fafafa' }}>
      <GradientMeshBg />

      {/* Hero Section — shown to everyone */}
      <div className="relative" style={{ zIndex: 1 }}>
        <div className="max-w-6xl mx-auto px-4 sm:px-6 pt-16 pb-20">
          <FadeIn className="text-center">
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-slate-900 tracking-tight mb-5">
              Learn AI by{' '}
              <TextMorph />
            </h1>
            <p className="text-base sm:text-lg text-blue-600/70 font-medium tracking-wide mb-3">
              The FPR Technique: Fundamentals, Practicals and Research
            </p>
            <p className="text-lg sm:text-xl text-slate-500 max-w-2xl mx-auto leading-relaxed mb-10">
              Master AI and ML through curated articles, hands-on Colab notebooks, and real-world case studies.
            </p>

            {/* Animated Stats */}
            <div className="flex items-center justify-center gap-8 sm:gap-12 text-sm">
              <div className="text-center">
                <div className="text-2xl font-bold text-slate-900">
                  <AnimatedCounter target={liveCourses.length} />
                </div>
                <div className="text-slate-400 text-xs mt-0.5">Courses</div>
              </div>
              <div className="w-px h-8 bg-slate-200" />
              <div className="text-center">
                <div className="text-2xl font-bold text-slate-900">
                  <AnimatedCounter target={totalNotebooks} suffix="+" />
                </div>
                <div className="text-slate-400 text-xs mt-0.5">Notebooks</div>
              </div>
              <div className="w-px h-8 bg-slate-200" />
              <div className="text-center">
                <div className="text-2xl font-bold text-slate-900">100%</div>
                <div className="text-slate-400 text-xs mt-0.5">Hands-on</div>
              </div>
            </div>

            {/* CTA for guests */}
            {!user && (
              <div className="mt-10 flex items-center justify-center gap-3">
                <a
                  href={`${process.env.NEXT_PUBLIC_VIZUARA_URL || 'https://vizuara.ai'}/auth/signup?redirect=${encodeURIComponent(process.env.NEXT_PUBLIC_PODS_CALLBACK_URL || 'https://pods.vizuara.ai/api/auth/session')}`}
                  className="inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white font-medium text-sm px-7 py-3 rounded-xl transition-colors shadow-lg shadow-blue-600/20"
                >
                  Get Started Free
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
                  </svg>
                </a>
                <a
                  href={`${process.env.NEXT_PUBLIC_VIZUARA_URL || 'https://vizuara.ai'}/auth/login?redirect=${encodeURIComponent(process.env.NEXT_PUBLIC_PODS_CALLBACK_URL || 'https://pods.vizuara.ai/api/auth/session')}`}
                  className="text-sm text-slate-500 hover:text-slate-900 font-medium transition-colors px-5 py-3 rounded-xl border border-slate-200 hover:border-slate-300"
                >
                  Log in
                </a>
              </div>
            )}
          </FadeIn>
        </div>
      </div>

      {/* ================================================================ */}
      {/*  GUEST VIEW — Marketing / teaser page (not logged in)            */}
      {/* ================================================================ */}
      {!user && (
        <div className="relative max-w-6xl mx-auto px-4 sm:px-6 pb-16 space-y-16" style={{ zIndex: 1 }}>

          {/* Course Showcase — real examples */}
          <FadeIn delay={0.1}>
            <div className="text-center mb-10">
              <h2 className="text-2xl sm:text-3xl font-bold text-slate-900 mb-2">
                See what you&apos;ll learn
              </h2>
              <p className="text-base text-slate-500 max-w-lg mx-auto">
                Here&apos;s a preview from three of our courses.
              </p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-5">
              {showcaseCourses.map((course) => (
                <div key={course.title} className="glass-card rounded-2xl overflow-hidden group flex flex-col">
                  <div className="relative h-44 overflow-hidden bg-slate-50">
                    <Image
                      src={course.hero}
                      alt={course.title}
                      fill
                      className="object-contain p-3 group-hover:scale-105 transition-transform duration-500"
                      sizes="(max-width: 640px) 90vw, 380px"
                      unoptimized
                    />
                  </div>
                  <div className="p-5 flex-1 flex flex-col gap-3">
                    <h3 className="font-bold text-slate-900 text-lg">{course.title}</h3>
                    <div>
                      <p className="text-xs font-semibold text-blue-600 uppercase tracking-wider mb-1">Article</p>
                      <p className="text-base text-slate-700 leading-relaxed">{course.article}</p>
                    </div>
                    {course.notebooks.length > 0 && (
                      <div>
                        <p className="text-xs font-semibold text-emerald-600 uppercase tracking-wider mb-1">
                          {course.notebooks.length} Notebook{course.notebooks.length > 1 ? 's' : ''}
                        </p>
                        <ul className="space-y-0.5">
                          {course.notebooks.map((nb) => (
                            <li key={nb} className="text-base text-slate-700 flex items-start gap-1.5">
                              <span className="text-emerald-500/60 mt-px">&#8250;</span>
                              {nb}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                    {course.caseStudy && (
                      <div className="mt-auto">
                        <p className="text-xs font-semibold text-amber-600 uppercase tracking-wider mb-1">Case Study</p>
                        <p className="text-base text-slate-700">{course.caseStudy}</p>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>

            <p className="text-center text-sm text-slate-400 mt-5">
              + {liveCourses.length - 3} more courses across diffusion models, reinforcement learning, RAG, agents, and more
            </p>
          </FadeIn>

          {/* Topics sneak peek */}
          <FadeIn delay={0.15}>
            <div className="text-center">
              <h2 className="text-2xl sm:text-3xl font-bold text-slate-900 mb-3">Topics we cover</h2>
              <p className="text-base text-slate-500 mb-6 max-w-lg mx-auto">
                New concepts added as soon as they emerge in the AI landscape &mdash; always taught from scratch.
              </p>
              <div className="flex flex-wrap items-center justify-center gap-2.5 max-w-2xl mx-auto">
                {topicPills.map((topic) => (
                  <span
                    key={topic}
                    className="px-4 py-2 rounded-full text-base text-slate-600 bg-white border border-slate-200"
                  >
                    {topic}
                  </span>
                ))}
                <span className="px-4 py-2 rounded-full text-base text-blue-600 bg-blue-50 border border-blue-200">
                  + {liveCourses.length - topicPills.length > 0 ? `${liveCourses.length - topicPills.length} more` : 'more coming'}
                </span>
              </div>
            </div>
          </FadeIn>

          <FoundersLetter />

          {/* How each course works — the FPR journey */}
          <FadeIn delay={0.2}>
            <div className="text-center mb-8">
              <h2 className="text-2xl sm:text-3xl font-bold text-slate-900 mb-2">Every course, three steps</h2>
              <p className="text-base text-slate-500 max-w-lg mx-auto">
                Our FPR framework gives you the right depth on every concept &mdash; not a 15-minute skim, not a 3-week deep dive.
              </p>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div className="glass-card rounded-2xl p-6 text-center">
                <div className="text-3xl font-bold text-blue-200 mb-2">01</div>
                <h3 className="font-bold text-slate-900 text-lg mb-2">Fundamentals</h3>
                <p className="text-base text-slate-700 leading-relaxed">
                  A comprehensive article with figures, equations, and worked examples explains each concept from first principles.
                </p>
              </div>
              <div className="glass-card rounded-2xl p-6 text-center">
                <div className="text-3xl font-bold text-emerald-200 mb-2">02</div>
                <h3 className="font-bold text-slate-900 text-lg mb-2">Practicals</h3>
                <p className="text-base text-slate-700 leading-relaxed">
                  Build every concept from scratch in Google Colab notebooks with guided TODO exercises and verification cells.
                </p>
              </div>
              <div className="glass-card rounded-2xl p-6 text-center">
                <div className="text-3xl font-bold text-amber-200 mb-2">03</div>
                <h3 className="font-bold text-slate-900 text-lg mb-2">Research</h3>
                <p className="text-base text-slate-700 leading-relaxed">
                  Tackle an industry case study that applies the concept to a real-world problem with production constraints.
                </p>
              </div>
            </div>
          </FadeIn>

          {/* Final CTA */}
          <FadeIn delay={0.25}>
            <div className="glass-card rounded-2xl p-10 text-center">
              <h2 className="text-2xl sm:text-3xl font-bold text-slate-900 mb-3">
                Ready to start building?
              </h2>
              <p className="text-slate-700 text-base mb-6 max-w-lg mx-auto">
                Join Vizuara AI Pods to access all courses, notebooks, case studies, and certificates.
              </p>
              <a
                href={`${process.env.NEXT_PUBLIC_VIZUARA_URL || 'https://vizuara.ai'}/auth/signup?redirect=${encodeURIComponent(process.env.NEXT_PUBLIC_PODS_CALLBACK_URL || 'https://pods.vizuara.ai/api/auth/session')}`}
                className="inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white font-medium text-sm px-7 py-3 rounded-xl transition-colors shadow-lg shadow-blue-600/20"
              >
                Sign Up Free
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
                </svg>
              </a>
            </div>
          </FadeIn>
        </div>
      )}

      {/* ================================================================ */}
      {/*  LOGGED-IN VIEW — Full course catalog with pods                  */}
      {/* ================================================================ */}
      {user && (
        <div className="relative max-w-6xl mx-auto px-4 sm:px-6 pb-16 space-y-12" style={{ zIndex: 1 }}>

          {/* Recommended for You */}
          {user.onboardingComplete && (
            <FadeIn delay={0.05}>
              {recommendedCourses.length > 0 ? (
                <div>
                  <DarkSectionHeader
                    title="Recommended for You"
                    subtitle="Based on your interests"
                  />
                  <CourseCarousel courses={recommendedCourses} completions={completions} />
                </div>
              ) : user.interests.length === 0 ? (
                <Link href="/profile#interests">
                  <div className="glass-card rounded-xl px-6 py-4 flex items-center gap-4 cursor-pointer group">
                    <div className="w-10 h-10 rounded-xl bg-blue-50 flex items-center justify-center flex-shrink-0">
                      <svg className="w-5 h-5 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
                      </svg>
                    </div>
                    <div>
                      <p className="text-sm font-medium text-slate-900">Tell us your interests to get personalized recommendations</p>
                      <p className="text-xs text-slate-500">Click here to select topics you care about</p>
                    </div>
                    <svg className="w-4 h-4 text-slate-400 ml-auto flex-shrink-0 group-hover:text-slate-900 transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
                    </svg>
                  </div>
                </Link>
              ) : (
                <div>
                  <DarkSectionHeader title="Recommended for You" />
                  <p className="text-sm text-slate-400">We&apos;re still learning your taste! Browse all courses below.</p>
                </div>
              )}
            </FadeIn>
          )}

          {/* Continue Learning */}
          {continueLearning.length > 0 && (
            <FadeIn delay={0.1}>
              <DarkSectionHeader
                title="Continue Learning"
                subtitle="Pick up where you left off"
              />
              <CourseCarousel courses={continueLearning} completions={completions} />
            </FadeIn>
          )}

          {/* All Courses with Pods */}
          <FadeIn delay={0.2}>
            <DarkSectionHeader title="All Courses" />
            <div className="flex items-center gap-4 mb-8 flex-wrap">
              <div className="flex-1 min-w-[200px] max-w-md">
                <SearchBar value={search} onChange={setSearch} />
              </div>
              <DifficultyFilter courses={liveCourses} onFilter={setBrowseFiltered} />
            </div>

            {searchedCourses.length === 0 ? (
              <div className="text-center py-16">
                <p className="text-slate-500 text-sm">No courses match your filter.</p>
              </div>
            ) : (
              <div className="space-y-10">
                {searchedCourses.map((course) => {
                  const livePods = coursePods[course.slug] || [];
                  const allPods = courseAllPods[course.slug] || [];
                  const liveSlugs = new Set(livePods.map((p) => p.slug));
                  const completion = completions[course.slug] || 0;

                  return (
                    <div key={course.slug}>
                      {/* Course header */}
                      <div className="flex items-start sm:items-center gap-3 mb-4 flex-wrap">
                        <Link href={`/courses/${course.slug}`} className="group">
                          <h3 className="text-xl font-bold text-slate-900 group-hover:text-blue-600 transition-colors">
                            {course.title}
                          </h3>
                        </Link>
                        <span className={`text-[11px] font-medium px-2 py-0.5 rounded-full border ${difficultyColor[course.difficulty]}`}>
                          {course.difficulty}
                        </span>
                        <span className="text-xs text-slate-400">~{course.estimatedHours}h</span>
                        <span className="text-xs text-slate-400">
                          {allPods.length} pod{allPods.length !== 1 ? 's' : ''}
                        </span>
                        {(course.totalNotebooks || 0) > 0 && (
                          <span className="text-xs text-slate-400 flex items-center gap-1">
                            <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                              <path strokeLinecap="round" strokeLinejoin="round" d="M17.25 6.75L22.5 12l-5.25 5.25m-10.5 0L1.5 12l5.25-5.25m7.5-3l-4.5 16.5" />
                            </svg>
                            {course.totalNotebooks} notebooks
                          </span>
                        )}
                        {completion > 0 && (
                          <span className="text-xs font-semibold text-emerald-600 ml-auto">
                            {completion}% complete
                          </span>
                        )}
                      </div>
                      <p className="text-base text-slate-500 mb-4 max-w-3xl">{course.description}</p>

                      {/* Course progress bar */}
                      {completion > 0 && (
                        <div className="mb-4 max-w-md">
                          <div className="w-full h-1 bg-slate-100 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-emerald-500 rounded-full transition-all duration-500"
                              style={{ width: `${completion}%` }}
                            />
                          </div>
                        </div>
                      )}

                      {/* Pod cards — horizontal scroll (live + coming soon) */}
                      {allPods.length > 0 ? (
                        <div className="flex gap-4 overflow-x-auto scrollbar-hide pb-2">
                          {allPods.map((pod) => {
                            const isLive = liveSlugs.has(pod.slug);

                            if (!isLive) {
                              // Coming soon pod — grayed out, no link
                              return (
                                <div
                                  key={pod.slug}
                                  className="flex-shrink-0 w-[280px] sm:w-[320px]"
                                >
                                  <div className="bg-white/60 rounded-2xl border border-slate-200/60 overflow-hidden h-full flex flex-col opacity-50">
                                    {pod.thumbnail ? (
                                      <div className="relative w-full aspect-[16/9] overflow-hidden bg-slate-50">
                                        <Image
                                          src={pod.thumbnail}
                                          alt={pod.title}
                                          fill
                                          className="object-cover"
                                          sizes="320px"
                                        />
                                        <div className="absolute inset-0 bg-gradient-to-t from-white via-white/20 to-transparent" />
                                        <div className="absolute top-2 left-2">
                                          <span className="bg-amber-100 text-amber-700 text-[10px] font-bold px-2 py-0.5 rounded-full border border-amber-200">
                                            Coming Soon
                                          </span>
                                        </div>
                                      </div>
                                    ) : (
                                      <div className="w-full aspect-[16/9] bg-gradient-to-br from-slate-100 to-slate-50 flex items-center justify-center relative">
                                        <span className="text-2xl font-bold text-slate-200">{pod.order}</span>
                                        <div className="absolute top-2 left-2">
                                          <span className="bg-amber-100 text-amber-700 text-[10px] font-bold px-2 py-0.5 rounded-full border border-amber-200">
                                            Coming Soon
                                          </span>
                                        </div>
                                      </div>
                                    )}
                                    <div className="p-4 flex flex-col flex-1">
                                      <h4 className="font-bold text-slate-500 text-lg leading-snug line-clamp-2 mb-1.5">
                                        {pod.title}
                                      </h4>
                                      <div className="flex items-center gap-2 mt-auto text-[11px] text-slate-300">
                                        <span>~{pod.estimatedHours}h</span>
                                      </div>
                                    </div>
                                  </div>
                                </div>
                              );
                            }

                            // Live pod — interactive with hover expansion
                            return (
                              <div
                                key={pod.slug}
                                className="flex-shrink-0 w-[280px] sm:w-[320px]"
                                onMouseEnter={() => handlePodMouseEnter(course.slug, course.title, pod)}
                                onMouseLeave={handlePodMouseLeave}
                              >
                                <Link href={`/courses/${course.slug}/${pod.slug}`}>
                                  <div className="bg-white rounded-2xl overflow-hidden h-full flex flex-col cursor-pointer group border border-blue-200 shadow-sm hover:shadow-lg hover:border-blue-300 hover:-translate-y-1 transition-all duration-300">
                                    {pod.thumbnail ? (
                                      <div className="relative w-full aspect-[16/9] overflow-hidden bg-slate-50">
                                        <Image
                                          src={pod.thumbnail}
                                          alt={pod.title}
                                          fill
                                          className="object-cover transition-transform duration-500 group-hover:scale-110"
                                          sizes="320px"
                                        />
                                        <div className="absolute inset-0 bg-gradient-to-t from-white via-white/20 to-transparent" />
                                      </div>
                                    ) : (
                                      <div className="w-full aspect-[16/9] bg-gradient-to-br from-blue-50 to-indigo-50 flex items-center justify-center">
                                        <span className="text-2xl font-bold text-blue-200">{pod.order}</span>
                                      </div>
                                    )}
                                    <div className="p-4 flex flex-col flex-1">
                                      <h4 className="font-bold text-slate-900 text-lg leading-snug mb-1.5 group-hover:text-blue-600 transition-colors line-clamp-2">
                                        {pod.title}
                                      </h4>
                                      <p className="text-base text-slate-700 leading-relaxed mb-3 line-clamp-2">
                                        {pod.description}
                                      </p>
                                      <div className="flex items-center gap-2 mt-auto pt-2 border-t border-slate-100">
                                        {pod.notebookCount > 0 && (
                                          <span className="text-[11px] text-slate-400 flex items-center gap-1">
                                            <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                              <path strokeLinecap="round" strokeLinejoin="round" d="M17.25 6.75L22.5 12l-5.25 5.25m-10.5 0L1.5 12l5.25-5.25m7.5-3l-4.5 16.5" />
                                            </svg>
                                            {pod.notebookCount} notebooks
                                          </span>
                                        )}
                                        {pod.hasCaseStudy && (
                                          <span className="text-[11px] text-slate-400 flex items-center gap-1">
                                            <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                              <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
                                            </svg>
                                            Case study
                                          </span>
                                        )}
                                        <span className="text-[11px] text-slate-400 ml-auto">~{pod.estimatedHours}h</span>
                                      </div>
                                    </div>
                                  </div>
                                </Link>
                              </div>
                            );
                          })}
                        </div>
                      ) : (
                        <p className="text-sm text-slate-400 italic">Pods coming soon</p>
                      )}

                      {/* Separator */}
                      <div className="mt-8 border-b border-slate-100" />
                    </div>
                  );
                })}
              </div>
            )}
          </FadeIn>

          {/* Coming Soon — aggregated upcoming pods + draft courses */}
          {(comingSoonPods.length > 0 || draftCourses.length > 0) && (
            <FadeIn delay={0.3}>
              <DarkSectionHeader
                title="Coming Soon"
                subtitle="New pods and courses in the pipeline"
              />

              {/* Draft courses */}
              {draftCourses.length > 0 && (
                <div className="mb-6">
                  <div className="flex gap-4 overflow-x-auto scrollbar-hide pb-2">
                    {draftCourses.map((course) => (
                      <div
                        key={course.slug}
                        className="flex-shrink-0 w-[280px] sm:w-[320px]"
                      >
                        <div className="bg-white/60 rounded-2xl border border-slate-200/60 overflow-hidden h-full flex flex-col opacity-70">
                          {course.thumbnail ? (
                            <div className="relative w-full aspect-[16/9] overflow-hidden bg-slate-50">
                              <Image
                                src={course.thumbnail}
                                alt={course.title}
                                fill
                                className="object-cover"
                                sizes="320px"
                              />
                              <div className="absolute inset-0 bg-gradient-to-t from-white via-white/20 to-transparent" />
                              <div className="absolute top-2.5 left-2.5">
                                <span className="bg-amber-50 backdrop-blur-sm border border-amber-200 rounded-full px-2 py-0.5 text-[11px] font-bold text-amber-700">
                                  Coming Soon
                                </span>
                              </div>
                            </div>
                          ) : (
                            <div className="relative w-full aspect-[16/9] bg-gradient-to-br from-slate-100 to-slate-50 flex items-center justify-center">
                              <svg className="w-8 h-8 text-slate-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                                <path strokeLinecap="round" strokeLinejoin="round" d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" />
                              </svg>
                              <div className="absolute top-2.5 left-2.5">
                                <span className="bg-amber-50 backdrop-blur-sm border border-amber-200 rounded-full px-2 py-0.5 text-[11px] font-bold text-amber-700">
                                  Coming Soon
                                </span>
                              </div>
                            </div>
                          )}
                          <div className="p-4 flex flex-col flex-1">
                            <h4 className="font-bold text-slate-700 text-lg leading-snug mb-1.5 line-clamp-2">
                              {course.title}
                            </h4>
                            <p className="text-base text-slate-400 leading-relaxed mb-3 line-clamp-2">{course.description}</p>
                            <div className="flex items-center gap-2 mt-auto pt-2 border-t border-slate-100">
                              <span className={`text-[11px] font-medium px-2 py-0.5 rounded-full border ${difficultyColor[course.difficulty]}`}>
                                {course.difficulty}
                              </span>
                              <span className="text-[11px] text-slate-400">{course.podCount} pod{course.podCount !== 1 ? 's' : ''} planned</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </FadeIn>
          )}

          {/* How each course works */}
          <FadeIn delay={0.35} className="mt-8 mb-8 max-w-3xl mx-auto">
            <h2 className="text-center text-base font-semibold text-slate-400 uppercase tracking-wider mb-8">
              How each course works
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div className="glass-card rounded-2xl p-6 text-center">
                <div className="text-3xl font-bold text-blue-200 mb-2">01</div>
                <h3 className="font-bold text-slate-900 text-lg mb-2">Fundamentals</h3>
                <p className="text-base text-slate-700 leading-relaxed">
                  Read an in-depth article with figures, equations, and worked examples.
                </p>
              </div>
              <div className="glass-card rounded-2xl p-6 text-center">
                <div className="text-3xl font-bold text-emerald-200 mb-2">02</div>
                <h3 className="font-bold text-slate-900 text-lg mb-2">Practicals</h3>
                <p className="text-base text-slate-700 leading-relaxed">
                  Build from scratch in Colab notebooks with guided TODO exercises.
                </p>
              </div>
              <div className="glass-card rounded-2xl p-6 text-center">
                <div className="text-3xl font-bold text-amber-200 mb-2">03</div>
                <h3 className="font-bold text-slate-900 text-lg mb-2">Research</h3>
                <p className="text-base text-slate-700 leading-relaxed">
                  Apply concepts to industry case studies with production constraints.
                </p>
              </div>
            </div>
          </FadeIn>
        </div>
      )}

      {/* Pod Expanded Card overlay — Prime Video style hover */}
      <AnimatePresence>
        {expandedPod && (
          <PodExpandedCard
            key={`${expandedPod.courseSlug}/${expandedPod.pod.slug}`}
            pod={expandedPod.pod}
            courseSlug={expandedPod.courseSlug}
            courseTitle={expandedPod.courseTitle}
            onClose={handleOverlayClose}
            onMouseEnter={handleOverlayMouseEnter}
            onMouseLeave={handleOverlayMouseLeave}
          />
        )}
      </AnimatePresence>
    </div>
  );
}
