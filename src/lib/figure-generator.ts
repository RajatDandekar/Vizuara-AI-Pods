/**
 * Figure generation using Google Gemini image generation API.
 * TypeScript port of generate_figure.py (Gemini path only — primary method).
 *
 * Uses @google/genai which is already installed in the project.
 */
import { GoogleGenAI } from '@google/genai';

/**
 * Generate a publication-quality figure using Gemini's image generation.
 * Returns the image data as a Buffer, or null if generation failed.
 */
export async function generateFigure(
  description: string
): Promise<Buffer | null> {
  const apiKey = process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error('GOOGLE_API_KEY or GEMINI_API_KEY not set');
  }

  const genai = new GoogleGenAI({ apiKey });

  const prompt = `Generate a clean, publication-ready technical diagram:

${description}

Style requirements:
- Clean white background
- Professional color palette (blues, teals, grays)
- Clear, readable labels with good font sizes
- No visual clutter or unnecessary decorations
- Suitable for a technical blog post (Substack)
- High contrast for readability on screens`;

  try {
    const response = await genai.models.generateContent({
      model: 'gemini-2.0-flash-exp',
      contents: prompt,
      config: {
        responseModalities: ['IMAGE', 'TEXT'],
      },
    });

    // Extract image data from the response
    if (response.candidates && response.candidates.length > 0) {
      const parts = response.candidates[0].content?.parts;
      if (parts) {
        for (const part of parts) {
          if (part.inlineData?.data) {
            return Buffer.from(part.inlineData.data, 'base64');
          }
        }
      }
    }

    console.log('Gemini returned no image data');
    return null;
  } catch (err) {
    console.error('Gemini figure generation failed:', err);
    return null;
  }
}
