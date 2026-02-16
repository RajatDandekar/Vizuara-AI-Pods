/**
 * ArXiv URL utilities and year-based visual styling for paper cards.
 */

export function extractArxivId(url: string): string | null {
  const match = url.match(/arxiv\.org\/(?:abs|pdf|html)\/(\d{4}\.\d{4,5})/);
  return match ? match[1] : null;
}

export function getArxivPdfUrl(arxivUrl: string): string | null {
  const id = extractArxivId(arxivUrl);
  if (!id) return null;
  return `https://arxiv.org/pdf/${id}`;
}

/** Returns Tailwind gradient classes based on paper year for visual variety. */
export function getYearGradient(year: number): { from: string; to: string } {
  if (year >= 2023) return { from: 'from-violet-600', to: 'to-purple-700' };
  if (year >= 2020) return { from: 'from-blue-600', to: 'to-indigo-700' };
  if (year >= 2017) return { from: 'from-cyan-600', to: 'to-blue-700' };
  if (year >= 2014) return { from: 'from-teal-600', to: 'to-emerald-700' };
  if (year >= 2010) return { from: 'from-amber-600', to: 'to-orange-700' };
  return { from: 'from-rose-600', to: 'to-pink-700' };
}
