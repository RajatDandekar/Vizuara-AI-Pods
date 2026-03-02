import { getDb } from './db';
import type {
  CreatorProject,
  CreateProjectRequest,
  Job,
  StepProgress,
  ProjectStep,
} from '@/types/creator';

function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

const DEFAULT_STEPS: StepProgress[] = [
  { step: 'article', status: 'pending' },
  { step: 'figures', status: 'pending' },
  { step: 'preview', status: 'pending' },
  { step: 'export', status: 'pending' },
  { step: 'notebooks', status: 'pending' },
  { step: 'case-study', status: 'pending' },
  { step: 'narration', status: 'pending' },
  { step: 'publish', status: 'pending' },
];

// ── Project CRUD ──────────────────────────────────────────────────────

export async function createProject(
  req: CreateProjectRequest,
  userId: string
): Promise<CreatorProject> {
  const id = generateId();
  const now = new Date().toISOString();

  const project: CreatorProject = {
    id,
    concept: req.concept,
    podTitle: req.podTitle,
    podSlug: req.podSlug,
    courseSlug: req.courseSlug,
    createdAt: now,
    updatedAt: now,
    createdBy: userId,
    currentStep: 'article',
    steps: DEFAULT_STEPS.map((s) => ({ ...s })),
    figures: [],
    notebooks: [],
  };

  const db = getDb();
  const { error } = await db
    .from('creator_projects')
    .insert({ id, data: project });

  if (error) throw new Error(`Failed to create project: ${error.message}`);
  return project;
}

export async function getProject(
  projectId: string
): Promise<CreatorProject | null> {
  const db = getDb();
  const { data, error } = await db
    .from('creator_projects')
    .select('data')
    .eq('id', projectId)
    .single();

  if (error || !data) return null;
  return data.data as CreatorProject;
}

export async function updateProject(
  projectId: string,
  patch: Partial<CreatorProject>
): Promise<CreatorProject> {
  const existing = await getProject(projectId);
  if (!existing) throw new Error(`Project not found: ${projectId}`);

  const updated: CreatorProject = {
    ...existing,
    ...patch,
    id: existing.id,
    updatedAt: new Date().toISOString(),
  };

  const db = getDb();
  const { error } = await db
    .from('creator_projects')
    .update({ data: updated, updated_at: new Date().toISOString() })
    .eq('id', projectId);

  if (error) throw new Error(`Failed to update project: ${error.message}`);
  return updated;
}

export async function deleteProject(projectId: string): Promise<void> {
  const db = getDb();

  // Delete all artifacts from storage
  const { data: files } = await db.storage
    .from('creator-artifacts')
    .list(projectId, { limit: 1000 });

  if (files && files.length > 0) {
    // List subdirectories and remove all files
    const allPaths: string[] = [];
    for (const file of files) {
      if (file.id) {
        // It's a file
        allPaths.push(`${projectId}/${file.name}`);
      } else {
        // It's a folder — list its contents
        const { data: subFiles } = await db.storage
          .from('creator-artifacts')
          .list(`${projectId}/${file.name}`, { limit: 1000 });
        if (subFiles) {
          for (const sf of subFiles) {
            allPaths.push(`${projectId}/${file.name}/${sf.name}`);
          }
        }
      }
    }
    if (allPaths.length > 0) {
      await db.storage.from('creator-artifacts').remove(allPaths);
    }
  }

  // Delete project row (cascade deletes jobs)
  const { error } = await db
    .from('creator_projects')
    .delete()
    .eq('id', projectId);

  if (error) throw new Error(`Failed to delete project: ${error.message}`);
}

export async function listProjects(): Promise<CreatorProject[]> {
  const db = getDb();
  const { data, error } = await db
    .from('creator_projects')
    .select('data')
    .order('updated_at', { ascending: false });

  if (error) {
    console.error('Failed to list projects:', error.message);
    return [];
  }

  return (data || []).map((row) => row.data as CreatorProject);
}

// ── Step management ───────────────────────────────────────────────────

export async function updateStepStatus(
  projectId: string,
  step: ProjectStep,
  status: StepProgress['status']
): Promise<void> {
  const project = await getProject(projectId);
  if (!project) throw new Error(`Project not found: ${projectId}`);

  project.steps = project.steps.map((s) =>
    s.step === step
      ? {
          ...s,
          status,
          completedAt: status === 'complete' ? new Date().toISOString() : s.completedAt,
        }
      : s
  );

  await updateProject(projectId, { steps: project.steps });
}

// ── Article helpers (no filesystem, stored in JSONB) ─────────────────

export async function saveOutline(
  projectId: string,
  outline: string
): Promise<void> {
  await updateProject(projectId, { outline, outlineApproved: false });
}

export async function saveArticleDraft(
  projectId: string,
  draft: string
): Promise<void> {
  await updateProject(projectId, { articleDraft: draft, articleApproved: false });
}

export async function getArticleDraft(
  projectId: string
): Promise<string | null> {
  const project = await getProject(projectId);
  return project?.articleDraft || null;
}

// ── Job system ───────────────────────────────────────────────────────

export async function createJob(
  projectId: string,
  type: Job['type']
): Promise<Job> {
  const id = generateId();
  const now = new Date().toISOString();

  const job: Job = {
    id,
    projectId,
    type,
    status: 'pending',
    createdAt: now,
    updatedAt: now,
  };

  const db = getDb();
  const { error } = await db
    .from('creator_jobs')
    .insert({ id, project_id: projectId, data: job });

  if (error) throw new Error(`Failed to create job: ${error.message}`);
  return job;
}

export async function getJob(
  projectId: string,
  jobId: string
): Promise<Job | null> {
  const db = getDb();
  const { data, error } = await db
    .from('creator_jobs')
    .select('data')
    .eq('id', jobId)
    .eq('project_id', projectId)
    .single();

  if (error || !data) return null;
  return data.data as Job;
}

export async function updateJob(
  projectId: string,
  jobId: string,
  patch: Partial<Job>
): Promise<void> {
  const job = await getJob(projectId, jobId);
  if (!job) throw new Error(`Job not found: ${jobId}`);

  const updated = { ...job, ...patch, updatedAt: new Date().toISOString() };

  const db = getDb();
  const { error } = await db
    .from('creator_jobs')
    .update({ data: updated, updated_at: new Date().toISOString() })
    .eq('id', jobId);

  if (error) throw new Error(`Failed to update job: ${error.message}`);
}

// ── Supabase Storage helpers ─────────────────────────────────────────

const BUCKET = 'creator-artifacts';

export async function uploadArtifact(
  projectId: string,
  subPath: string,
  data: Buffer | Blob,
  contentType: string
): Promise<string> {
  const db = getDb();
  const fullPath = `${projectId}/${subPath}`;

  const { error } = await db.storage
    .from(BUCKET)
    .upload(fullPath, data, {
      contentType,
      upsert: true,
    });

  if (error) throw new Error(`Failed to upload artifact: ${error.message}`);
  return fullPath;
}

export async function downloadArtifact(
  projectId: string,
  subPath: string
): Promise<Buffer> {
  const db = getDb();
  const fullPath = `${projectId}/${subPath}`;

  const { data, error } = await db.storage
    .from(BUCKET)
    .download(fullPath);

  if (error || !data) {
    throw new Error(`Failed to download artifact: ${error?.message || 'Not found'}`);
  }

  return Buffer.from(await data.arrayBuffer());
}

export async function getArtifactUrl(
  projectId: string,
  subPath: string,
  expiresIn: number = 3600
): Promise<string> {
  const db = getDb();
  const fullPath = `${projectId}/${subPath}`;

  const { data, error } = await db.storage
    .from(BUCKET)
    .createSignedUrl(fullPath, expiresIn);

  if (error || !data?.signedUrl) {
    throw new Error(`Failed to get artifact URL: ${error?.message || 'Unknown error'}`);
  }

  return data.signedUrl;
}

// ── Utility ───────────────────────────────────────────────────────────

/**
 * Parse figure placeholders from article draft.
 * Format: {{FIGURE: description || caption}}
 */
export function parseFigurePlaceholders(
  draft: string
): Array<{ id: string; description: string; caption: string }> {
  const regex = /\{\{FIGURE:\s*(.+?)\s*(?:\|\|\s*(.+?))?\s*\}\}/g;
  const figures: Array<{ id: string; description: string; caption: string }> =
    [];
  let match;
  let index = 1;

  while ((match = regex.exec(draft)) !== null) {
    figures.push({
      id: `figure_${index}`,
      description: match[1].trim(),
      caption: match[2]?.trim() || match[1].trim(),
    });
    index++;
  }

  return figures;
}
