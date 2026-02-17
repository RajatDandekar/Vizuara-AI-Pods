import { NextRequest } from 'next/server';
import { GoogleGenAI } from '@google/genai';
import { sleep, getRetryDelay, isRateLimitError } from '@/lib/retry';

export const dynamic = 'force-dynamic';

const CHAT_MODEL = 'gemini-2.5-flash';
const MAX_RETRIES = 3;

const SYSTEM_PROMPT = `You are an AI teaching assistant for the Vizuara learning platform. You have access to the full content of a Google Colab notebook that a student is currently working through.

Your role:
- Help students understand concepts, code, and exercises in this notebook
- Explain things clearly using simple language
- Reference specific parts of the notebook when relevant
- Provide code examples or fixes when helpful
- Be encouraging and educational

Guidelines:
- Keep answers concise but thorough (2-4 paragraphs typical)
- Use markdown formatting for clarity
- For code questions, show corrected code with brief explanations
- For concept questions, build from what the notebook already teaches
- If a question is unrelated to the notebook, briefly answer but redirect to the notebook content
- When referencing math, use LaTeX notation ($...$)

NOTEBOOK CONTENT:
`;

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

  const { question, context, history = [] } = body as {
    question: string;
    context: string;
    history: { role: string; content: string }[];
  };

  if (!question || !context) {
    return new Response(
      JSON.stringify({ error: 'question and context are required' }),
      { status: 400, headers: { 'Content-Type': 'application/json' } }
    );
  }

  if (!process.env.GEMINI_API_KEY) {
    return new Response(
      JSON.stringify({ error: 'API key not configured' }),
      { status: 503, headers: { 'Content-Type': 'application/json' } }
    );
  }

  const genai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
  const encoder = new TextEncoder();
  const truncatedContext = context.slice(0, 100_000);

  // Build multi-turn conversation
  const contents: { role: string; parts: { text: string }[] }[] = [];
  for (const msg of history) {
    contents.push({
      role: msg.role === 'user' ? 'user' : 'model',
      parts: [{ text: msg.content }],
    });
  }
  contents.push({ role: 'user', parts: [{ text: question }] });

  const readableStream = new ReadableStream({
    async start(controller) {
      for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
        try {
          const response = await genai.models.generateContentStream({
            model: CHAT_MODEL,
            contents,
            config: {
              systemInstruction: SYSTEM_PROMPT + truncatedContext,
              temperature: 0.7,
              maxOutputTokens: 2048,
            },
          });

          for await (const chunk of response) {
            const text = chunk.text;
            if (text) {
              const data = JSON.stringify({ type: 'text', content: text });
              controller.enqueue(encoder.encode(`data: ${data}\n\n`));
            }
          }

          controller.enqueue(
            encoder.encode(`data: ${JSON.stringify({ type: 'done' })}\n\n`)
          );
          controller.close();
          return;
        } catch (error) {
          if (isRateLimitError(error) && attempt < MAX_RETRIES) {
            const delay = getRetryDelay(error);
            console.log(`Chat stream rate limited. Retrying in ${delay}ms (attempt ${attempt + 1})...`);
            controller.enqueue(
              encoder.encode(`data: ${JSON.stringify({ type: 'text', content: '\n\n*Waiting for API rate limit... retrying shortly.*\n\n' })}\n\n`)
            );
            await sleep(delay);
            continue;
          }
          const message = error instanceof Error ? error.message : 'Unknown error';
          controller.enqueue(
            encoder.encode(`data: ${JSON.stringify({ type: 'error', message })}\n\n`)
          );
          controller.close();
          return;
        }
      }
      controller.enqueue(
        encoder.encode(`data: ${JSON.stringify({ type: 'error', message: 'Rate limit exceeded after retries. Please wait a minute and try again.' })}\n\n`)
      );
      controller.close();
    },
  });

  return new Response(readableStream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive',
    },
  });
}
