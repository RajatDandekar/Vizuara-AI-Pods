import { GoogleGenAI } from '@google/genai';

let _genai: GoogleGenAI | null = null;

export function getGenAI(): GoogleGenAI {
  if (!_genai) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error(
        'GEMINI_API_KEY is not set. Please create a .env.local file with your API key.'
      );
    }
    _genai = new GoogleGenAI({ apiKey });
  }
  return _genai;
}
