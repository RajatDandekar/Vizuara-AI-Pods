import Anthropic from '@anthropic-ai/sdk';

const MAX_RETRIES = 3;

function sleep(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function getRetryDelay(attempt: number): number {
  // Exponential backoff: 5s, 10s, 20s
  return Math.min(5000 * Math.pow(2, attempt), 30000);
}

/**
 * Create an SSE streaming Response using the Anthropic SDK.
 * Mirrors the pattern from stream.ts but uses Claude instead of Gemini.
 */
export function createClaudeSSEStream(
  systemPrompt: string,
  userPrompt: string,
  options?: { maxTokens?: number; model?: string }
): Response {
  const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY || '' });
  const encoder = new TextEncoder();
  const model = options?.model || 'claude-sonnet-4-20250514';
  const maxTokens = options?.maxTokens || 16000;

  const readableStream = new ReadableStream({
    async start(controller) {
      for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
        try {
          const stream = client.messages.stream({
            model,
            max_tokens: maxTokens,
            system: systemPrompt,
            messages: [{ role: 'user', content: userPrompt }],
          });

          for await (const event of stream) {
            if (
              event.type === 'content_block_delta' &&
              event.delta.type === 'text_delta'
            ) {
              const text = event.delta.text;
              if (text) {
                const data = JSON.stringify({ type: 'text', content: text });
                controller.enqueue(encoder.encode(`data: ${data}\n\n`));
              }
            }
          }

          controller.enqueue(
            encoder.encode(`data: ${JSON.stringify({ type: 'done' })}\n\n`)
          );
          controller.close();
          return;
        } catch (error) {
          const isRateLimit =
            error instanceof Anthropic.RateLimitError ||
            (error instanceof Error && error.message.includes('rate_limit'));

          if (isRateLimit && attempt < MAX_RETRIES) {
            const delay = getRetryDelay(attempt);
            console.log(
              `Claude stream rate limited. Retrying in ${delay}ms (attempt ${attempt + 1})...`
            );
            controller.enqueue(
              encoder.encode(
                `data: ${JSON.stringify({
                  type: 'text',
                  content:
                    '\n\n*Waiting for API rate limit... retrying shortly.*\n\n',
                })}\n\n`
              )
            );
            await sleep(delay);
            continue;
          }

          const message =
            error instanceof Error ? error.message : 'Unknown error';
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
          `data: ${JSON.stringify({
            type: 'error',
            message:
              'Rate limit exceeded after retries. Please wait a minute and try again.',
          })}\n\n`
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

/**
 * Create an SSE stream that also accumulates text for post-processing.
 * The onComplete callback receives the full text after streaming ends.
 */
export function createClaudeSSEStreamWithAccumulation(
  systemPrompt: string,
  userPrompt: string,
  onComplete: (fullText: string) => Promise<void>,
  options?: { maxTokens?: number; model?: string }
): Response {
  const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY || '' });
  const encoder = new TextEncoder();
  const model = options?.model || 'claude-sonnet-4-20250514';
  const maxTokens = options?.maxTokens || 16000;
  let fullText = '';

  const readableStream = new ReadableStream({
    async start(controller) {
      for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
        try {
          const stream = client.messages.stream({
            model,
            max_tokens: maxTokens,
            system: systemPrompt,
            messages: [{ role: 'user', content: userPrompt }],
          });

          for await (const event of stream) {
            if (
              event.type === 'content_block_delta' &&
              event.delta.type === 'text_delta'
            ) {
              const text = event.delta.text;
              if (text) {
                fullText += text;
                const data = JSON.stringify({ type: 'text', content: text });
                controller.enqueue(encoder.encode(`data: ${data}\n\n`));
              }
            }
          }

          // Run post-processing
          try {
            await onComplete(fullText);
          } catch (err) {
            console.error('Post-processing error:', err);
          }

          controller.enqueue(
            encoder.encode(`data: ${JSON.stringify({ type: 'done' })}\n\n`)
          );
          controller.close();
          return;
        } catch (error) {
          const isRateLimit =
            error instanceof Anthropic.RateLimitError ||
            (error instanceof Error && error.message.includes('rate_limit'));

          if (isRateLimit && attempt < MAX_RETRIES) {
            const delay = getRetryDelay(attempt);
            controller.enqueue(
              encoder.encode(
                `data: ${JSON.stringify({
                  type: 'text',
                  content:
                    '\n\n*Waiting for API rate limit... retrying shortly.*\n\n',
                })}\n\n`
              )
            );
            await sleep(delay);
            continue;
          }

          const message =
            error instanceof Error ? error.message : 'Unknown error';
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
          `data: ${JSON.stringify({
            type: 'error',
            message:
              'Rate limit exceeded after retries. Please wait a minute and try again.',
          })}\n\n`
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

/**
 * Non-streaming Claude call for short tasks (classify, plan, etc.)
 */
export async function claudeComplete(
  systemPrompt: string,
  userPrompt: string,
  options?: { maxTokens?: number; model?: string }
): Promise<string> {
  const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY || '' });
  const model = options?.model || 'claude-sonnet-4-20250514';
  const maxTokens = options?.maxTokens || 4096;

  const response = await client.messages.create({
    model,
    max_tokens: maxTokens,
    system: systemPrompt,
    messages: [{ role: 'user', content: userPrompt }],
  });

  const textBlock = response.content.find((b) => b.type === 'text');
  return textBlock?.text || '';
}
