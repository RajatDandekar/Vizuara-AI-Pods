import { NextRequest } from 'next/server';
import { GoogleGenAI } from '@google/genai';

export const dynamic = 'force-dynamic';

const CHAT_MODEL = 'gemini-2.5-flash';

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

const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
};

export async function POST(request: NextRequest) {
  let body: Record<string, unknown>;
  try {
    body = await request.json();
  } catch {
    return new Response(
      JSON.stringify({ error: 'Invalid or empty request body' }),
      { status: 400, headers: { 'Content-Type': 'application/json', ...CORS_HEADERS } }
    );
  }

  try {
    const { question, context, history = [] } = body as {
      question: string;
      context: string;
      history: { role: string; content: string }[];
    };

    if (!question || !context) {
      return new Response(
        JSON.stringify({ error: 'question and context are required' }),
        { status: 400, headers: { 'Content-Type': 'application/json', ...CORS_HEADERS } }
      );
    }

    if (!process.env.GEMINI_API_KEY) {
      return new Response(
        JSON.stringify({ error: 'API key not configured' }),
        { status: 503, headers: { 'Content-Type': 'application/json', ...CORS_HEADERS } }
      );
    }

    const genai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

    // Truncate context to stay within token limits
    const truncatedContext = context.slice(0, 100000);

    // Build multi-turn conversation
    const contents: { role: string; parts: { text: string }[] }[] = [];
    for (const msg of (history as { role: string; content: string }[])) {
      contents.push({
        role: msg.role === 'user' ? 'user' : 'model',
        parts: [{ text: msg.content }],
      });
    }
    contents.push({
      role: 'user',
      parts: [{ text: question }],
    });

    const response = await genai.models.generateContent({
      model: CHAT_MODEL,
      contents,
      config: {
        systemInstruction: SYSTEM_PROMPT + truncatedContext,
        temperature: 0.7,
        maxOutputTokens: 2048,
      },
    });

    const answer = response.text || 'I could not generate a response. Please try again.';

    return new Response(
      JSON.stringify({ answer }),
      { status: 200, headers: { 'Content-Type': 'application/json', ...CORS_HEADERS } }
    );
  } catch (error) {
    console.error('Chat error:', error);
    return new Response(
      JSON.stringify({ error: 'Failed to generate response' }),
      { status: 500, headers: { 'Content-Type': 'application/json', ...CORS_HEADERS } }
    );
  }
}

export async function OPTIONS() {
  return new Response(null, { status: 204, headers: CORS_HEADERS });
}
