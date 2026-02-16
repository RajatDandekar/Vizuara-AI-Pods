'use client';

import { motion } from 'framer-motion';
import { useMemo } from 'react';

const COLORS = ['#d97706', '#2563eb', '#16a34a', '#e11d48', '#f59e0b', '#8b5cf6'];
const PIECE_COUNT = 60;

export default function Confetti() {
  const pieces = useMemo(
    () =>
      Array.from({ length: PIECE_COUNT }, (_, i) => ({
        id: i,
        x: Math.random() * 100,
        delay: Math.random() * 0.5,
        duration: 2 + Math.random() * 2,
        color: COLORS[Math.floor(Math.random() * COLORS.length)],
        rotation: Math.random() * 360,
        size: 6 + Math.random() * 6,
        shape: Math.random() > 0.5 ? ('circle' as const) : ('rect' as const),
      })),
    []
  );

  return (
    <div className="fixed inset-0 pointer-events-none z-50 overflow-hidden">
      {pieces.map((p) => (
        <motion.div
          key={p.id}
          initial={{
            x: `${p.x}vw`,
            y: '-10px',
            rotate: 0,
            opacity: 1,
          }}
          animate={{
            y: '110vh',
            rotate: p.rotation + 360,
            opacity: [1, 1, 0],
          }}
          transition={{
            duration: p.duration,
            delay: p.delay,
            ease: 'easeIn',
          }}
          style={{
            position: 'absolute',
            width: p.size,
            height: p.shape === 'circle' ? p.size : p.size * 0.6,
            borderRadius: p.shape === 'circle' ? '50%' : '2px',
            backgroundColor: p.color,
          }}
        />
      ))}
    </div>
  );
}
