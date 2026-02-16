'use client';

import Image from 'next/image';
import Link from 'next/link';
import { motion } from 'framer-motion';
import FadeIn from '@/components/animations/FadeIn';

export default function FoundersLetter() {
  return (
    <FadeIn delay={0.1}>
      <div className="glass-card rounded-2xl p-8 sm:p-12 relative overflow-hidden">
        {/* Subtle accent glow */}
        <div className="absolute top-0 right-0 w-64 h-64 bg-blue-500/5 rounded-full blur-3xl pointer-events-none" />

        <p className="text-base font-medium text-blue-600 tracking-wider uppercase mb-6">
          A Letter from the Founders
        </p>

        <h2 className="text-3xl sm:text-4xl font-bold text-slate-900 mb-8 leading-tight max-w-2xl">
          Why we built VizFlix
        </h2>

        <div className="space-y-5 text-slate-700 leading-relaxed text-base sm:text-lg max-w-3xl">
          <p>
            The field of AI is evolving at a pace unlike anything we have seen before.
            New concepts emerge every week &mdash; just look at 2025 alone: DeepSeek
            changed the game for open reasoning models, a wave of new reasoning architectures
            followed, AI agents went from research demos to production systems, and world
            models opened entirely new frontiers. Keeping up is hard. Really understanding
            these ideas from a foundational level? Even harder.
          </p>

          <p>
            The traditional path &mdash; reading papers, decoding the math, implementing the
            theory, and truly understanding the nuts and bolts of each concept &mdash; takes
            enormous time and effort. Most people either settle for shallow 15-minute video
            tutorials that skip the fundamentals, or spend weeks on a single paper and fall
            behind on everything else.
          </p>

          <p className="text-slate-900 font-medium">
            VizFlix was built to solve exactly this problem.
          </p>

          <p>
            Our platform cuts through the noise and presents every concept in the most
            fundamental way possible. Students understand the foundations behind each idea,
            build them from scratch in hands-on notebooks, and then apply them to real-world
            case studies to build genuine confidence. You don&apos;t spend too much time on
            each concept, and you don&apos;t spend too little &mdash; it&apos;s the right
            amount of depth, designed to make the learning stick.
          </p>

          <p>
            This is something we learned during our PhDs at MIT and Purdue. The best learning
            happens when rigor meets relevance &mdash; when you understand the math deeply
            enough to implement it, and when you implement it in a context that actually
            matters. We have built this platform with that philosophy at its core.
          </p>

          <p>
            Many students who joined our bootcamps at Vizuara told us the same thing: they
            wanted a single place where new AI concepts are added as they emerge, taught
            from scratch, without having to pay for each topic individually. VizFlix is
            that place &mdash; a subscription-based platform that stays up to date with
            the evolving landscape of AI, delivering each concept through our{' '}
            <span className="text-blue-600 font-medium">FPR framework</span>: Fundamentals,
            Practicals, and Research.
          </p>

          <p>
            As always, we teach everything for beginners, completely from scratch.
          </p>
        </div>

        {/* Founders */}
        <div className="mt-10 pt-8 border-t border-slate-100">
          <div className="flex flex-wrap items-center gap-6">
            {[
              { name: 'Dr Raj Dandekar', image: '/founders/raj.png' },
              { name: 'Dr Rajat Dandekar', image: '/founders/rajat.jpg' },
              { name: 'Dr Sreedath Panat', image: '/founders/sreedath.jpg' },
            ].map((founder) => (
              <div key={founder.name} className="flex items-center gap-3">
                <div className="relative w-10 h-10 rounded-full overflow-hidden ring-1 ring-slate-200">
                  <Image
                    src={founder.image}
                    alt={founder.name}
                    fill
                    className="object-cover"
                    sizes="40px"
                    unoptimized
                  />
                </div>
                <span className="text-base text-slate-700">{founder.name}</span>
              </div>
            ))}
          </div>
          <p className="text-sm text-slate-400 mt-4">
            Founders, Vizuara &mdash; IIT Madras, MIT, Purdue
          </p>
        </div>
      </div>
    </FadeIn>
  );
}
