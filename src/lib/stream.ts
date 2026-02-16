import { GoogleGenAI } from '@google/genai';
import { MODEL_NAME } from './constants';
import { sleep, getRetryDelay, isRateLimitError } from './retry';

const MAX_RETRIES = 3;

export function createSSEStream(
  prompt: string
): Response {
  const genai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || '' });
  const encoder = new TextEncoder();

  const readableStream = new ReadableStream({
    async start(controller) {
      for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
        try {
          const response = await genai.models.generateContentStream({
            model: MODEL_NAME,
            contents: prompt,
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
            console.log(`Stream rate limited. Retrying in ${delay}ms (attempt ${attempt + 1})...`);
            controller.enqueue(
              encoder.encode(`data: ${JSON.stringify({ type: 'text', content: '\n\n*Waiting for API rate limit... retrying shortly.*\n\n' })}\n\n`)
            );
            await sleep(delay);
            continue;
          }
          const message = error instanceof Error ? error.message : 'Unknown error';
          controller.enqueue(
            encoder.encode(
              `data: ${JSON.stringify({ type: 'error', message })}\n\n`
            )
          );
          controller.close();
          return;
        }
      }
      controller.enqueue(
        encoder.encode(
          `data: ${JSON.stringify({ type: 'error', message: 'Rate limit exceeded after retries. Please wait a minute and try again.' })}\n\n`
        )
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

export function createNotebookSSEStream(
  prompt: string,
  buildNotebook: (text: string) => unknown
): Response {
  const genai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || '' });
  const encoder = new TextEncoder();
  let fullText = '';

  const readableStream = new ReadableStream({
    async start(controller) {
      for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
        try {
          const response = await genai.models.generateContentStream({
            model: MODEL_NAME,
            contents: prompt,
          });

          for await (const chunk of response) {
            const text = chunk.text;
            if (text) {
              fullText += text;
              const data = JSON.stringify({ type: 'text', content: text });
              controller.enqueue(encoder.encode(`data: ${data}\n\n`));
            }
          }

          try {
            const notebook = buildNotebook(fullText);
            controller.enqueue(
              encoder.encode(
                `data: ${JSON.stringify({ type: 'notebook', content: notebook })}\n\n`
              )
            );
          } catch {
            // Notebook parse failed, still signal done
          }

          controller.enqueue(
            encoder.encode(`data: ${JSON.stringify({ type: 'done' })}\n\n`)
          );
          controller.close();
          return;
        } catch (error) {
          if (isRateLimitError(error) && attempt < MAX_RETRIES) {
            const delay = getRetryDelay(error);
            console.log(`Notebook stream rate limited. Retrying in ${delay}ms (attempt ${attempt + 1})...`);
            controller.enqueue(
              encoder.encode(`data: ${JSON.stringify({ type: 'text', content: '\n\n*Waiting for API rate limit... retrying shortly.*\n\n' })}\n\n`)
            );
            await sleep(delay);
            continue;
          }
          const message = error instanceof Error ? error.message : 'Unknown error';
          controller.enqueue(
            encoder.encode(
              `data: ${JSON.stringify({ type: 'error', message })}\n\n`
            )
          );
          controller.close();
          return;
        }
      }
      controller.enqueue(
        encoder.encode(
          `data: ${JSON.stringify({ type: 'error', message: 'Rate limit exceeded after retries. Please wait a minute and try again.' })}\n\n`
        )
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
