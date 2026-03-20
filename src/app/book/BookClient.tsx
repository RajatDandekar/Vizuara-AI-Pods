'use client';

import Image from 'next/image';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { useAuth } from '@/context/AuthContext';
import { useSubscription } from '@/context/SubscriptionContext';
import FadeIn from '@/components/animations/FadeIn';
import GradientMeshBg from '@/components/home/GradientMeshBg';

const VIZUARA_URL = process.env.NEXT_PUBLIC_VIZUARA_URL || 'https://vizuara.ai';
const PODS_CALLBACK_URL = process.env.NEXT_PUBLIC_PODS_CALLBACK_URL || 'https://pods.vizuara.ai/api/auth/session';

const stats = [
  { label: 'Chapters', value: '13' },
  { label: 'Topics', value: '73' },
  { label: 'Diagrams', value: '200+' },
  { label: 'Pages', value: '400+' },
];

const chapters = [
  {
    number: 1,
    title: 'Build a Neural Network from Scratch',
    topics: ['Perceptrons', 'Activation Functions', 'Backpropagation', 'Loss Functions', 'Gradient Descent', 'Neural Network from Scratch'],
    free: true,
  },
  {
    number: 2,
    title: 'Convolutional Neural Networks',
    topics: ['Convolution Operation', 'Pooling Layers', 'CNN Architectures', 'Feature Maps', 'Image Classification'],
  },
  {
    number: 3,
    title: 'Recurrent Neural Networks & Sequence Models',
    topics: ['RNNs', 'LSTMs', 'GRUs', 'Sequence-to-Sequence', 'Bidirectional RNNs'],
  },
  {
    number: 4,
    title: 'The Transformer Architecture',
    topics: ['Self-Attention', 'Multi-Head Attention', 'Positional Encoding', 'Encoder-Decoder', 'Layer Normalization'],
  },
  {
    number: 5,
    title: 'Large Language Models',
    topics: ['GPT Architecture', 'Tokenization', 'Pre-training', 'Scaling Laws', 'Emergent Abilities'],
  },
  {
    number: 6,
    title: 'Training & Fine-Tuning LLMs',
    topics: ['RLHF', 'LoRA', 'QLoRA', 'Instruction Tuning', 'DPO', 'Supervised Fine-Tuning'],
  },
  {
    number: 7,
    title: 'Prompt Engineering & In-Context Learning',
    topics: ['Zero-Shot', 'Few-Shot', 'Chain-of-Thought', 'ReAct', 'Prompt Chaining'],
  },
  {
    number: 8,
    title: 'Retrieval-Augmented Generation (RAG)',
    topics: ['Vector Databases', 'Embedding Models', 'Chunking Strategies', 'Hybrid Search', 'RAG Pipelines'],
  },
  {
    number: 9,
    title: 'AI Agents & Tool Use',
    topics: ['Agent Architectures', 'Tool Calling', 'Planning', 'Memory Systems', 'Multi-Agent Systems'],
  },
  {
    number: 10,
    title: 'Diffusion Models & Image Generation',
    topics: ['Denoising', 'DDPM', 'Stable Diffusion', 'ControlNet', 'Image-to-Image'],
  },
  {
    number: 11,
    title: 'Reinforcement Learning',
    topics: ['MDPs', 'Q-Learning', 'Policy Gradients', 'PPO', 'RLHF Connection'],
  },
  {
    number: 12,
    title: 'Multimodal AI',
    topics: ['Vision-Language Models', 'CLIP', 'Visual Question Answering', 'Image Captioning'],
  },
  {
    number: 13,
    title: 'Deploying AI to Production',
    topics: ['Model Serving', 'Quantization', 'Distillation', 'Edge Deployment', 'Monitoring'],
  },
];

const benefitCards = [
  {
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178z" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
      </svg>
    ),
    title: 'Visual Learning',
    desc: 'Every concept explained with hand-crafted diagrams — no walls of text.',
  },
  {
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M4.26 10.147a60.438 60.438 0 00-.491 6.347A48.62 48.62 0 0112 20.904a48.62 48.62 0 018.232-4.41 60.46 60.46 0 00-.491-6.347m-15.482 0a50.636 50.636 0 00-2.658-.813A59.906 59.906 0 0112 3.493a59.903 59.903 0 0110.399 5.84c-.896.248-1.783.52-2.658.814m-15.482 0A50.717 50.717 0 0112 13.489a50.702 50.702 0 017.74-3.342" />
      </svg>
    ),
    title: 'From Scratch',
    desc: 'Build up from perceptrons to production LLMs — no prerequisites assumed.',
  },
  {
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z" />
      </svg>
    ),
    title: 'Cutting Edge',
    desc: 'Covers the latest — Transformers, Diffusion Models, AI Agents, RAG, and more.',
  },
  {
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" />
      </svg>
    ),
    title: 'Reference Guide',
    desc: '400+ pages you can revisit anytime — a lasting resource for your AI journey.',
  },
];

export default function BookClient() {
  const { user } = useAuth();
  const { enrolled, loading } = useSubscription();

  return (
    <>
      <GradientMeshBg />
      <div className="relative z-10 max-w-5xl mx-auto px-4 sm:px-6 py-12 sm:py-16">
        {/* Hero */}
        <FadeIn>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 items-center mb-16">
            {/* Book cover */}
            <div className="flex justify-center">
              <div style={{ perspective: '1200px' }}>
                <motion.div
                  style={{
                    transformStyle: 'preserve-3d',
                    transform: 'rotateY(-12deg)',
                  }}
                  whileHover={{ transform: 'rotateY(-4deg)' }}
                  transition={{ duration: 0.4, ease: [0.25, 0.1, 0.25, 1] as [number, number, number, number] }}
                >
                  <div
                    className="absolute top-0 left-0 h-full w-4 bg-gradient-to-b from-indigo-700 to-violet-800 rounded-l-sm"
                    style={{
                      transform: 'rotateY(90deg) translateZ(8px) translateX(-8px)',
                      transformOrigin: 'left center',
                    }}
                  />
                  <div
                    className="absolute -bottom-4 left-4 right-0 h-6 rounded-xl"
                    style={{
                      background: 'radial-gradient(ellipse at center, rgba(99,102,241,0.25) 0%, transparent 70%)',
                      filter: 'blur(10px)',
                    }}
                  />
                  <div className="relative w-[260px] sm:w-[300px] aspect-[3/4] rounded-lg overflow-hidden shadow-2xl border border-indigo-200/40">
                    <Image
                      src="/book/cover.png"
                      alt="Vizuara: A Complete Visual Guide to Mastering Modern AI and LLMs"
                      fill
                      className="object-cover"
                      sizes="300px"
                      priority
                    />
                  </div>
                </motion.div>
              </div>
            </div>

            {/* Info */}
            <div>
              <h1 className="text-3xl sm:text-4xl font-bold text-slate-900 mb-3">
                Vizuara: A Complete Visual Guide to AI
              </h1>
              <p className="text-lg text-slate-500 mb-6">
                Master modern AI and LLMs through 200+ hand-crafted diagrams — from neural networks to production systems.
              </p>

              {/* Stats */}
              <div className="flex flex-wrap gap-6 mb-6">
                {stats.map((s) => (
                  <div key={s.label} className="text-center">
                    <div className="text-2xl font-bold text-indigo-700">{s.value}</div>
                    <div className="text-xs text-slate-400">{s.label}</div>
                  </div>
                ))}
              </div>

              <DownloadCTA user={user} enrolled={enrolled} loading={loading} />
            </div>
          </div>
        </FadeIn>

        {/* Table of Contents */}
        <FadeIn delay={0.1}>
          <div className="mb-16">
            <h2 className="text-2xl font-bold text-slate-900 mb-2 text-center">Full Table of Contents</h2>
            <p className="text-sm text-slate-500 text-center mb-8">13 chapters covering the full modern AI stack</p>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {chapters.map((ch) => (
                <div
                  key={ch.number}
                  className="relative bg-white/80 backdrop-blur-sm border border-slate-200/60 rounded-xl p-5 hover:shadow-md transition-shadow"
                >
                  {ch.free && (
                    <span className="absolute top-3 right-3 px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 text-[10px] font-semibold border border-emerald-200">
                      Free Preview
                    </span>
                  )}
                  <div className="text-xs font-medium text-indigo-500 mb-1">Chapter {ch.number}</div>
                  <h3 className="text-sm font-semibold text-slate-900 mb-3">{ch.title}</h3>
                  <div className="flex flex-wrap gap-1.5">
                    {ch.topics.map((t) => (
                      <span key={t} className="px-2 py-0.5 rounded-full text-[11px] bg-slate-50 text-slate-500 border border-slate-100">
                        {t}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </FadeIn>

        {/* Sample Preview */}
        <FadeIn delay={0.15}>
          <div className="mb-16">
            <h2 className="text-2xl font-bold text-slate-900 mb-2 text-center">Sample Pages from Chapter 1</h2>
            <p className="text-sm text-slate-500 text-center mb-8">See how every concept is explained visually</p>

            <div className="flex justify-center">
              <div className="relative h-[240px] sm:h-[320px] w-full max-w-lg">
                {[1, 2, 3].map((n, i) => (
                  <motion.div
                    key={n}
                    className="absolute top-0 left-1/2 w-[180px] sm:w-[240px] aspect-[3/4] rounded-xl overflow-hidden shadow-lg border border-slate-200 bg-white"
                    style={{
                      transform: `translateX(-50%) rotate(${(i - 1) * 6}deg) translateX(${(i - 1) * 50}px)`,
                      zIndex: 3 - i,
                    }}
                    whileHover={{ y: -12, scale: 1.03 }}
                    transition={{ duration: 0.25 }}
                  >
                    <Image
                      src={`/book/sample-${n}.png`}
                      alt={`Sample page ${n}`}
                      fill
                      className="object-cover"
                      sizes="240px"
                    />
                  </motion.div>
                ))}
              </div>
            </div>
          </div>
        </FadeIn>

        {/* Benefits Grid */}
        <FadeIn delay={0.2}>
          <div className="mb-16">
            <h2 className="text-2xl font-bold text-slate-900 mb-8 text-center">Why This Book</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {benefitCards.map((b) => (
                <div
                  key={b.title}
                  className="bg-white/80 backdrop-blur-sm border border-slate-200/60 rounded-xl p-6 hover:shadow-md transition-shadow"
                >
                  <div className="text-indigo-600 mb-3">{b.icon}</div>
                  <h3 className="text-sm font-semibold text-slate-900 mb-1">{b.title}</h3>
                  <p className="text-sm text-slate-500">{b.desc}</p>
                </div>
              ))}
            </div>
          </div>
        </FadeIn>

        {/* Download CTA */}
        <FadeIn delay={0.25}>
          <div className="text-center mb-16 bg-gradient-to-br from-indigo-50 to-violet-50 rounded-2xl border border-indigo-200/60 p-8">
            <h2 className="text-xl font-bold text-slate-900 mb-2">Ready to learn visually?</h2>
            <p className="text-sm text-slate-500 mb-6">Get the full 400+ page book as a downloadable PDF.</p>
            <DownloadCTA user={user} enrolled={enrolled} loading={loading} />
          </div>
        </FadeIn>

        {/* Cross-sell */}
        <FadeIn delay={0.3}>
          <div className="text-center">
            <p className="text-sm text-slate-500 mb-2">Want hands-on practice?</p>
            <Link
              href="/"
              className="text-sm font-medium text-indigo-600 hover:text-indigo-800 transition-colors"
            >
              Explore our interactive courses →
            </Link>
          </div>
        </FadeIn>
      </div>
    </>
  );
}

function DownloadCTA({
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
      <div className="flex justify-center">
        <div className="w-6 h-6 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  if (enrolled) {
    return (
      <a
        href="/api/book/download"
        className="inline-flex items-center gap-2 px-7 py-3.5 bg-indigo-600 text-white font-medium rounded-xl hover:bg-indigo-700 transition-colors shadow-sm text-base"
      >
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
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
        className="inline-flex items-center gap-2 px-7 py-3.5 bg-indigo-600 text-white font-medium rounded-xl hover:bg-indigo-700 transition-colors shadow-sm text-base"
      >
        Subscribe to Access
      </a>
    );
  }

  return (
    <a
      href={`${VIZUARA_URL}/auth/signup?redirect=${encodeURIComponent(PODS_CALLBACK_URL)}`}
      className="inline-flex items-center gap-2 px-7 py-3.5 bg-indigo-600 text-white font-medium rounded-xl hover:bg-indigo-700 transition-colors shadow-sm text-base"
    >
      Sign Up to Access
    </a>
  );
}
