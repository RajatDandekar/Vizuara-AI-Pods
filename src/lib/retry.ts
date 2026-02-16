/**
 * Waits for the specified number of milliseconds.
 */
export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Extracts retry delay from a Gemini 429 error message, or returns a default.
 */
export function getRetryDelay(error: unknown): number {
  const message = error instanceof Error ? error.message : String(error);
  // Look for "retry in Xs" or "retryDelay":"Xs" patterns
  const match = message.match(/retry.*?(\d+(?:\.\d+)?)\s*s/i);
  if (match) {
    return Math.ceil(parseFloat(match[1]) * 1000) + 500; // Add 500ms buffer
  }
  return 6000; // Default 6 seconds
}

/**
 * Check if an error is a rate limit (429) error.
 */
export function isRateLimitError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error);
  return message.includes('429') || message.includes('RESOURCE_EXHAUSTED') || message.includes('quota');
}
