import type { Metadata } from 'next';
import BookClient from './BookClient';

export const metadata: Metadata = {
  title: 'Visual AI Book — Vizuara AI Pods',
  description:
    'Vizuara: A Complete Visual Guide to Mastering Modern AI and LLMs. 13 chapters, 73 topics, 200+ diagrams, 400+ pages.',
};

export default function BookPage() {
  return <BookClient />;
}
