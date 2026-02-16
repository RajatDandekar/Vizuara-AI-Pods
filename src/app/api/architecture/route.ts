import { NextRequest } from 'next/server';
import { paperArchitecturePrompt } from '@/lib/prompts';
import { createSSEStream } from '@/lib/stream';

export const dynamic = 'force-dynamic';

export async function POST(request: NextRequest) {
  let body: Record<string, unknown>;
  try {
    body = await request.json();
  } catch {
    return new Response(
      JSON.stringify({ error: 'Invalid or empty request body' }),
      { status: 400, headers: { 'Content-Type': 'application/json' } }
    );
  }

  try {
    const { concept, paper } = body;

    if (!concept || !(paper as Record<string, unknown>)?.title) {
      return new Response(
        JSON.stringify({ error: 'concept and paper are required' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      );
    }

    const p = paper as Record<string, unknown>;
    const prompt = paperArchitecturePrompt(concept as string, {
      title: p.title as string,
      authors: (p.authors as string[]) || [],
      year: (p.year as number) || 2000,
      venue: (p.venue as string) || '',
    });
    return createSSEStream(prompt);
  } catch (error) {
    console.error('Architecture streaming error:', error);
    return new Response(
      JSON.stringify({ error: 'Failed to generate architecture analysis' }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    );
  }
}
