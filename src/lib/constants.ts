export const MODEL_NAME = 'gemini-2.5-pro';
export const NUM_PAPERS = 5;

export const SUGGESTED_CONCEPTS = [
  'Transformers',
  'Generative Adversarial Networks',
  'Diffusion Models',
  'Attention Mechanism',
  'Reinforcement Learning from Human Feedback',
  'Convolutional Neural Networks',
  'Variational Autoencoders',
  'Graph Neural Networks',
];

// Pods that are fully free — accessible without login or subscription
export const FREE_POD_SLUGS = new Set(['intro-ddpm', 'basics-of-rl', 'vision-encoders']);

// Pods that are free but require login (no subscription needed)
export const FREE_WITH_LOGIN_POD_SLUGS = new Set(['openclaw-rl']);

export const FREE_POD_SPECS = [
  { courseSlug: 'diffusion-models', podSlug: 'intro-ddpm' },
  { courseSlug: 'rl-from-scratch', podSlug: 'basics-of-rl' },
  { courseSlug: 'vlms-from-scratch', podSlug: 'vision-encoders' },
  { courseSlug: 'rl-from-scratch', podSlug: 'openclaw-rl' },
] as const;
