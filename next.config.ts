import type { NextConfig } from "next";

/**
 * Legacy URL redirects.
 * Old flat structure: /courses/{podSlug}/...
 * New hierarchical:   /courses/{courseSlug}/{podSlug}/...
 */
const LEGACY_REDIRECTS: { old: string; course: string }[] = [
  { old: '5d-parallelism-from-scratch', course: 'gpu-programming' },
  { old: 'context-engineering-for-llms', course: 'context-engineering' },
  { old: 'diffusion-llms-from-scratch', course: 'build-diffusion-llm' },
  { old: 'diffusion-models-video-generation', course: 'diffusion-models' },
  { old: 'tiny-recursive-models', course: 'tiny-recursive-models' },
  { old: 'understanding-bert-from-scratch', course: 'build-llm' },
  { old: 'vision-transformers-from-scratch', course: 'vlms-from-scratch' },
  { old: 'vla-autonomous-driving', course: 'vlas-autonomous-driving' },
  { old: 'world-action-models', course: 'modern-robot-learning' },
  { old: 'world-models', course: 'modern-robot-learning' },
];

const nextConfig: NextConfig = {
  outputFileTracingExcludes: {
    '/*': [
      './public/courses/**',
      './public/notebooks/**',
      './public/case-studies/**',
      './public/narration/**',
      './public/founders/**',
      './public/showcase/**',
    ],
  },
  async redirects() {
    const rules: {
      source: string;
      destination: string;
      permanent: boolean;
    }[] = [];

    for (const { old, course } of LEGACY_REDIRECTS) {
      // Skip self-referencing redirects (where course slug == pod slug)
      // These would cause infinite redirect loops since the new URL structure
      // /courses/{courseSlug}/{podSlug} matches the old /courses/{podSlug}
      // when courseSlug === podSlug
      if (old === course) continue;

      // Redirect /courses/{old} → /courses/{course}/{old}
      rules.push({
        source: `/courses/${old}`,
        destination: `/courses/${course}/${old}`,
        permanent: true,
      });
      // Redirect /courses/{old}/article → /courses/{course}/{old}/article (etc.)
      rules.push({
        source: `/courses/${old}/:path*`,
        destination: `/courses/${course}/${old}/:path*`,
        permanent: true,
      });
    }

    return rules;
  },
};

export default nextConfig;
