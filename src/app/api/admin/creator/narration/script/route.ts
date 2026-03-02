import { NextRequest, NextResponse } from 'next/server';
import { getAdminUser } from '@/lib/admin';
import { getProject, updateProject } from '@/lib/creator-projects';
import { claudeComplete } from '@/lib/claude-stream';
import { getNarrationScriptPrompt } from '@/lib/creator-prompts';
import type { NarrationSegment } from '@/types/creator';

export const dynamic = 'force-dynamic';
export const maxDuration = 120;

interface RawSegment {
  segment_id?: string;
  cell_indices?: number[];
  insert_before?: string;
  narration_text?: string;
  text?: string;
  type?: string;
  duration_estimate_seconds?: number;
  estimatedDuration?: number;
}

export async function POST(request: NextRequest) {
  const admin = await getAdminUser();
  if (!admin) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  const { projectId, notebookId } = await request.json();
  const project = await getProject(projectId);
  if (!project) return NextResponse.json({ error: 'Project not found' }, { status: 404 });

  // Read the notebook content from JSONB (no filesystem)
  const notebook = project.notebooks.find((n) => n.id === notebookId);
  if (!notebook?.markdownContent) {
    return NextResponse.json({ error: 'Notebook content not found' }, { status: 404 });
  }

  const result = await claudeComplete(
    'You are an expert educator creating narration scripts for Colab notebooks. Output only valid JSON.',
    getNarrationScriptPrompt(notebook.markdownContent),
    { maxTokens: 8000 }
  );

  // Parse JSON
  const jsonMatch = result.match(/\[[\s\S]*\]/);
  if (!jsonMatch) {
    return NextResponse.json({ error: 'Failed to parse narration script' }, { status: 500 });
  }

  const rawSegments: RawSegment[] = JSON.parse(jsonMatch[0]);

  // Map to NarrationSegment format
  const segments: NarrationSegment[] = rawSegments.map((s, i) => {
    const claudeSegId = s.segment_id || `seg_${String(i).padStart(2, '0')}`;
    const segId = claudeSegId.startsWith(notebookId)
      ? claudeSegId
      : `${notebookId}_${claudeSegId}`;
    const text = s.narration_text || s.text || '';
    const cellIndices = s.cell_indices || (s.insert_before ? [] : []);
    const cellIndex = cellIndices.length > 0 ? cellIndices[0] : i;
    const duration = s.duration_estimate_seconds || s.estimatedDuration || 45;

    return {
      id: segId,
      segment_id: segId,
      notebookId,
      cellIndex,
      cell_indices: cellIndices,
      insert_before: s.insert_before || '',
      text,
      narration_text: text,
      duration,
      type: s.type || (i === 0 ? 'intro' : 'explanation'),
    } as NarrationSegment;
  });

  // Update project (no filesystem writes)
  const existingNarration = project.narration || {
    status: 'script-ready' as const,
    segments: [],
  };

  await updateProject(projectId, {
    narration: {
      ...existingNarration,
      status: 'script-ready',
      segments: [
        ...existingNarration.segments.filter(
          (s: NarrationSegment) => s.notebookId !== notebookId
        ),
        ...segments,
      ],
    },
  });

  return NextResponse.json({ segments });
}
