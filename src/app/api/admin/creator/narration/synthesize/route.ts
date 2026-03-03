import { NextRequest, NextResponse } from 'next/server';
import { getAdminUser } from '@/lib/admin';
import { getProject, updateProject, uploadArtifact } from '@/lib/creator-projects';
import type { NarrationSegment } from '@/types/creator';

export const dynamic = 'force-dynamic';
export const maxDuration = 300;

const ELEVENLABS_VOICE_ID = process.env.ELEVENLABS_VOICE_ID || 'lZORFNDokoBmfd0S06vf';

export async function POST(request: NextRequest) {
  const admin = await getAdminUser();
  if (!admin) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  const { projectId, notebookId } = await request.json();
  const project = await getProject(projectId);
  if (!project) return NextResponse.json({ error: 'Project not found' }, { status: 404 });

  const narration = project.narration;
  if (!narration) {
    return NextResponse.json({ error: 'No narration script found' }, { status: 400 });
  }

  const segments = narration.segments.filter(
    (s: NarrationSegment) => s.notebookId === notebookId
  );
  if (segments.length === 0) {
    return NextResponse.json({ error: 'No segments for this notebook' }, { status: 400 });
  }

  const apiKey = process.env.ELEVENLABS_API_KEY;
  if (!apiKey) {
    return NextResponse.json({ error: 'ELEVENLABS_API_KEY not configured' }, { status: 500 });
  }

  // Stream progress via SSE
  const encoder = new TextEncoder();
  const readableStream = new ReadableStream({
    async start(controller) {
      const sendProgress = (step: string, progress: number) => {
        controller.enqueue(
          encoder.encode(
            `data: ${JSON.stringify({ type: 'progress', step, progress })}\n\n`
          )
        );
      };

      try {
        for (let i = 0; i < segments.length; i++) {
          const segment = segments[i];
          sendProgress(
            `Synthesizing segment ${i + 1} of ${segments.length}...`,
            Math.round(((i) / segments.length) * 100)
          );

          // Call ElevenLabs TTS API
          const ttsRes = await fetch(
            `https://api.elevenlabs.io/v1/text-to-speech/${ELEVENLABS_VOICE_ID}`,
            {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                'xi-api-key': apiKey,
              },
              body: JSON.stringify({
                text: segment.text,
                model_id: 'eleven_multilingual_v2',
                voice_settings: {
                  stability: 0.5,
                  similarity_boost: 0.85,
                  style: 0.3,
                  use_speaker_boost: true,
                },
              }),
            }
          );

          if (!ttsRes.ok) {
            throw new Error(`ElevenLabs API error: ${ttsRes.status}`);
          }

          const audioBuffer = Buffer.from(await ttsRes.arrayBuffer());

          // Upload audio to Supabase Storage
          const storagePath = await uploadArtifact(
            projectId,
            `narration/${segment.id}.mp3`,
            audioBuffer,
            'audio/mpeg'
          );

          // Update segment with storage path
          segment.audioStoragePath = storagePath;
        }

        sendProgress('All segments synthesized', 100);

        // Update project narration state
        await updateProject(projectId, {
          narration: {
            ...narration,
            status: 'complete',
            segments: narration.segments.map((s: NarrationSegment) => {
              const updated = segments.find((u: NarrationSegment) => u.id === s.id);
              return updated || s;
            }),
          },
        });

        controller.enqueue(
          encoder.encode(`data: ${JSON.stringify({ type: 'done' })}\n\n`)
        );
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Synthesis failed';
        controller.enqueue(
          encoder.encode(
            `data: ${JSON.stringify({ type: 'error', message })}\n\n`
          )
        );
      }

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
