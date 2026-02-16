import { NextRequest, NextResponse } from 'next/server';
import { getGenAI } from '@/lib/anthropic';
import { paperDiscoveryPrompt } from '@/lib/prompts';
import { MODEL_NAME } from '@/lib/constants';
import { sleep, getRetryDelay, isRateLimitError } from '@/lib/retry';

export const dynamic = 'force-dynamic';

function extractJSON(text: string): string {
  const jsonMatch = text.match(/\{[\s\S]*\}/);
  if (jsonMatch) {
    return jsonMatch[0];
  }
  return text.trim();
}

async function callGeminiWithRetry(prompt: string, maxRetries = 3) {
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const response = await getGenAI().models.generateContent({
        model: MODEL_NAME,
        contents: prompt,
      });
      return response;
    } catch (error) {
      if (isRateLimitError(error) && attempt < maxRetries) {
        const delay = getRetryDelay(error);
        console.log(`Rate limited. Retrying in ${delay}ms (attempt ${attempt + 1}/${maxRetries})...`);
        await sleep(delay);
        continue;
      }
      throw error;
    }
  }
  throw new Error('Max retries exceeded');
}

export async function POST(request: NextRequest) {
  try {
    const { concept } = await request.json();

    if (!concept || typeof concept !== 'string') {
      return NextResponse.json(
        { error: 'A concept string is required' },
        { status: 400 }
      );
    }

    const response = await callGeminiWithRetry(paperDiscoveryPrompt(concept.trim()));

    const text = response.text ?? '';
    console.log('Gemini raw response length:', text.length);

    const jsonStr = extractJSON(text);
    const parsed = JSON.parse(jsonStr);

    if (!parsed.papers || !Array.isArray(parsed.papers)) {
      throw new SyntaxError('Response missing papers array');
    }

    const papers = parsed.papers.map((p: Record<string, unknown>, i: number) => ({
      id: p.id || `paper-${i}`,
      title: p.title || 'Unknown Paper',
      authors: Array.isArray(p.authors) ? p.authors : ['Unknown'],
      year: typeof p.year === 'number' ? p.year : 2000,
      venue: p.venue || 'Unknown Venue',
      arxivUrl: p.arxivUrl || '',
      oneLiner: p.oneLiner || '',
      significance: p.significance || '',
    }));

    return NextResponse.json({ concept: parsed.concept || concept, papers });
  } catch (error) {
    console.error('Paper discovery error:', error);

    if (error instanceof SyntaxError) {
      return NextResponse.json(
        { error: 'Failed to parse AI response. Please try again.' },
        { status: 502 }
      );
    }

    if (isRateLimitError(error)) {
      return NextResponse.json(
        { error: 'Rate limit reached. Please wait a moment and try again.' },
        { status: 429 }
      );
    }

    return NextResponse.json(
      { error: 'Failed to discover papers. Please check your API key and try again.' },
      { status: 500 }
    );
  }
}
