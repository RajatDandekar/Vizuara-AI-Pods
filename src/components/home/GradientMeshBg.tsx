'use client';

import { motion } from 'framer-motion';

export default function GradientMeshBg({ variant = 'light' }: { variant?: 'light' | 'dark' }) {
  const dark = variant === 'dark';

  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none" style={{ zIndex: 0 }}>
      {/* Base background */}
      <div className="absolute inset-0" style={{ background: dark ? '#020617' : '#fafafa' }} />

      {/* Soft orbs */}
      <motion.div
        className="absolute w-[600px] h-[600px] rounded-full"
        style={{
          background: dark
            ? 'radial-gradient(circle, rgba(59,130,246,0.12) 0%, transparent 70%)'
            : 'radial-gradient(circle, rgba(59,130,246,0.15) 0%, transparent 70%)',
          opacity: dark ? 0.5 : 0.3,
          top: '-10%',
          right: '10%',
        }}
        animate={{
          x: [0, 40, -20, 0],
          y: [0, -30, 20, 0],
        }}
        transition={{
          duration: 20,
          repeat: Infinity,
          ease: 'linear',
        }}
      />
      <motion.div
        className="absolute w-[500px] h-[500px] rounded-full"
        style={{
          background: dark
            ? 'radial-gradient(circle, rgba(99,102,241,0.10) 0%, transparent 70%)'
            : 'radial-gradient(circle, rgba(99,102,241,0.12) 0%, transparent 70%)',
          opacity: dark ? 0.4 : 0.25,
          bottom: '10%',
          left: '-5%',
        }}
        animate={{
          x: [0, -30, 30, 0],
          y: [0, 40, -20, 0],
        }}
        transition={{
          duration: 25,
          repeat: Infinity,
          ease: 'linear',
        }}
      />
      <motion.div
        className="absolute w-[400px] h-[400px] rounded-full"
        style={{
          background: dark
            ? 'radial-gradient(circle, rgba(168,85,247,0.08) 0%, transparent 70%)'
            : 'radial-gradient(circle, rgba(168,85,247,0.10) 0%, transparent 70%)',
          opacity: dark ? 0.35 : 0.2,
          top: '40%',
          left: '40%',
        }}
        animate={{
          x: [0, 50, -30, 0],
          y: [0, -40, 30, 0],
        }}
        transition={{
          duration: 30,
          repeat: Infinity,
          ease: 'linear',
        }}
      />

      {/* Subtle dot pattern */}
      <div
        className="absolute inset-0"
        style={{
          opacity: dark ? 0.06 : 0.04,
          backgroundImage: dark
            ? 'radial-gradient(circle at 1px 1px, rgba(255,255,255,0.2) 1px, transparent 0)'
            : 'radial-gradient(circle at 1px 1px, rgba(0,0,0,0.3) 1px, transparent 0)',
          backgroundSize: '40px 40px',
        }}
      />
    </div>
  );
}
