import type { NextConfig } from "next";

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
};

export default nextConfig;
