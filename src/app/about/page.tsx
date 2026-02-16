'use client';

import Image from 'next/image';
import Link from 'next/link';
import { motion } from 'framer-motion';
import FadeIn from '@/components/animations/FadeIn';
import GradientMeshBg from '@/components/home/GradientMeshBg';

const founders = [
  {
    name: 'Dr Raj Dandekar',
    image: '/founders/raj.png',
    initials: 'RD',
  },
  {
    name: 'Dr Rajat Dandekar',
    image: '/founders/rajat.png',
    initials: 'RD',
  },
  {
    name: 'Dr Sreedath Panat',
    image: '/founders/sreedath.png',
    initials: 'SP',
  },
];

const values = [
  {
    title: 'Excellence over shortcuts',
    description: 'Real skills, real outcomes.',
  },
  {
    title: 'Builder mindset',
    description: 'Learn by creating, shipping, and iterating.',
  },
  {
    title: 'Clarity and honesty',
    description: 'In feedback, curriculum, and expectations.',
  },
  {
    title: 'Student-first',
    description: 'Design every experience for learner success.',
  },
];

const quickFacts = [
  'Backed by the MIT ecosystem; built by IIT Madras, MIT, and Purdue alumni',
  'Curriculum designed with a "learn-by-building" philosophy',
  'Hands-on mentorship and feedback loops for faster growth',
];

const whatWeDo = [
  'Project-based AI, ML, and Generative AI programs with mentorship',
  'University application guidance and profile-building support',
  'Industry-relevant capstones, papers, and portfolio curation',
  'Career enablement \u2014 demos, reviews, and interview preparation',
];

export default function AboutPage() {
  return (
    <div className="light-landing min-h-screen" style={{ background: '#fafafa' }}>
      <GradientMeshBg />

      <div className="relative" style={{ zIndex: 1 }}>
        <div className="max-w-4xl mx-auto px-4 sm:px-6 pt-16 pb-24">

          {/* Header */}
          <FadeIn className="text-center mb-20">
            <p className="text-base font-medium text-blue-600 tracking-wider uppercase mb-4">About Us</p>
            <h1 className="text-4xl sm:text-5xl font-bold text-slate-900 tracking-tight mb-6">
              Making world-class AI education{' '}
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-cyan-500">
                accessible to everyone
              </span>
            </h1>
            <p className="text-xl text-slate-600 max-w-2xl mx-auto leading-relaxed">
              We are Vizuara &mdash; a fast-growing Indian startup backed by the MIT ecosystem,
              revolutionizing AI education for students and professionals.
            </p>
          </FadeIn>

          {/* Overview */}
          <FadeIn delay={0.1} className="mb-20">
            <div className="glass-card rounded-2xl p-8 sm:p-10">
              <p className="text-lg text-slate-700 leading-relaxed mb-4">
                Vizuara is founded by alumni from <span className="text-slate-900 font-semibold">IIT Madras</span>,{' '}
                <span className="text-slate-900 font-semibold">MIT</span>, and{' '}
                <span className="text-slate-900 font-semibold">Purdue University</span>. We blend academic rigor
                with practical, industry-aligned projects to help learners build real skills, faster.
              </p>
              <p className="text-lg text-slate-500 leading-relaxed">
                Starting with small cohorts and research-driven bootcamps, we saw how learners thrive when
                rigor meets relevance. Today, Vizuara brings that high-touch model to a broader audience
                &mdash; without compromising on depth, expectations, or outcomes.
              </p>
            </div>
          </FadeIn>

          {/* Mission & Vision */}
          <FadeIn delay={0.15} className="mb-20">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="glass-card rounded-2xl p-8">
                <div className="w-10 h-10 rounded-xl bg-blue-100 flex items-center justify-center mb-4">
                  <svg className="w-5 h-5 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15.59 14.37a6 6 0 01-5.84 7.38v-4.8m5.84-2.58a14.98 14.98 0 006.16-12.12A14.98 14.98 0 009.631 8.41m5.96 5.96a14.926 14.926 0 01-5.841 2.58m-.119-8.54a6 6 0 00-7.381 5.84h4.8m2.581-5.84a14.927 14.927 0 00-2.58 5.841m2.699 2.7c-.103.021-.207.041-.311.06a15.09 15.09 0 01-2.448-2.448 14.9 14.9 0 01.06-.312m-2.24 2.39a4.493 4.493 0 00-1.757 4.306 4.493 4.493 0 004.306-1.758M16.5 9a1.5 1.5 0 11-3 0 1.5 1.5 0 013 0z" />
                  </svg>
                </div>
                <h3 className="text-xl font-semibold text-slate-900 mb-2">Our Mission</h3>
                <p className="text-base text-slate-600 leading-relaxed">
                  Democratize high-quality AI education through structured programs, actionable projects,
                  and mentorship &mdash; enabling learners to go from fundamentals to real-world impact.
                </p>
              </div>
              <div className="glass-card rounded-2xl p-8">
                <div className="w-10 h-10 rounded-xl bg-purple-100 flex items-center justify-center mb-4">
                  <svg className="w-5 h-5 text-purple-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178z" />
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                </div>
                <h3 className="text-xl font-semibold text-slate-900 mb-2">Our Vision</h3>
                <p className="text-base text-slate-600 leading-relaxed">
                  To become the most trusted platform for AI education in emerging markets &mdash; where
                  ambitious learners build portfolios, publish work, and land meaningful opportunities.
                </p>
              </div>
            </div>
          </FadeIn>

          {/* What We Do */}
          <FadeIn delay={0.2} className="mb-20">
            <h2 className="text-2xl font-bold text-slate-900 mb-6">What We Do</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {whatWeDo.map((item) => (
                <div key={item} className="glass-card rounded-xl px-5 py-4 flex items-start gap-3">
                  <svg className="w-5 h-5 text-emerald-500 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <p className="text-base text-slate-700 leading-relaxed">{item}</p>
                </div>
              ))}
            </div>
          </FadeIn>

          {/* Quick Facts */}
          <FadeIn delay={0.25} className="mb-20">
            <h2 className="text-2xl font-bold text-slate-900 mb-6">Quick Facts</h2>
            <div className="space-y-3">
              {quickFacts.map((fact) => (
                <div key={fact} className="glass-card rounded-xl px-5 py-4 flex items-start gap-3">
                  <svg className="w-5 h-5 text-blue-500 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
                  </svg>
                  <p className="text-base text-slate-700 leading-relaxed">{fact}</p>
                </div>
              ))}
            </div>
          </FadeIn>

          {/* Founders */}
          <FadeIn delay={0.3} className="mb-20">
            <h2 className="text-2xl font-bold text-slate-900 mb-2">Our Founders</h2>
            <p className="text-base text-slate-500 mb-8">
              Alumni from IIT Madras, MIT, and Purdue University.
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-5">
              {founders.map((founder) => (
                <motion.div
                  key={founder.name}
                  className="glass-card rounded-2xl p-6 text-center group"
                  whileHover={{ y: -4 }}
                  transition={{ duration: 0.2 }}
                >
                  <div className="relative w-28 h-28 mx-auto mb-4 rounded-full overflow-hidden ring-2 ring-slate-200 group-hover:ring-blue-400 transition-all">
                    <Image
                      src={founder.image}
                      alt={founder.name}
                      fill
                      className="object-cover"
                      sizes="112px"
                    />
                  </div>
                  <h3 className="font-semibold text-slate-900 text-base">{founder.name}</h3>
                </motion.div>
              ))}
            </div>
          </FadeIn>

          {/* YouTube / Vizuara Channel */}
          <FadeIn delay={0.35} className="mb-20">
            <div className="glass-card rounded-2xl p-8 sm:p-10 flex flex-col sm:flex-row items-start gap-6">
              <div className="w-14 h-14 rounded-2xl bg-red-100 flex items-center justify-center flex-shrink-0">
                <svg className="w-7 h-7 text-red-500" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/>
                </svg>
              </div>
              <div>
                <h3 className="text-xl font-semibold text-slate-900 mb-2">Vizuara on YouTube</h3>
                <p className="text-base text-slate-600 leading-relaxed mb-4">
                  Our YouTube channel has grown significantly, reaching learners across the globe.
                  Through in-depth video explanations, walkthroughs, and tutorials, the channel
                  makes Vizuara&apos;s AI education accessible to everyone &mdash; for free.
                </p>
                <a
                  href="https://www.youtube.com/@vizuara"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 text-base text-red-500 hover:text-red-600 font-medium transition-colors"
                >
                  Visit the channel
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 6H5.25A2.25 2.25 0 003 8.25v10.5A2.25 2.25 0 005.25 21h10.5A2.25 2.25 0 0018 18.75V10.5m-10.5 6L21 3m0 0h-5.25M21 3v5.25" />
                  </svg>
                </a>
              </div>
            </div>
          </FadeIn>

          {/* Values */}
          <FadeIn delay={0.4} className="mb-12">
            <h2 className="text-2xl font-bold text-slate-900 mb-6">Our Values</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {values.map((value) => (
                <div key={value.title} className="glass-card rounded-xl p-6">
                  <h3 className="text-slate-900 font-semibold text-base mb-1">{value.title}</h3>
                  <p className="text-slate-500 text-base">{value.description}</p>
                </div>
              ))}
            </div>
          </FadeIn>

          {/* CTA */}
          <FadeIn delay={0.45} className="text-center">
            <div className="glass-card rounded-2xl p-10">
              <h2 className="text-3xl font-bold text-slate-900 mb-3">Ready to start learning?</h2>
              <p className="text-base text-slate-500 mb-6 max-w-md mx-auto">
                Explore our courses and start building real AI projects today.
              </p>
              <Link
                href="/"
                className="inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white font-medium text-base px-6 py-2.5 rounded-xl transition-colors"
              >
                Browse Courses
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
                </svg>
              </Link>
            </div>
          </FadeIn>
        </div>
      </div>
    </div>
  );
}
