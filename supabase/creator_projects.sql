-- Pod Creator: Supabase schema for project + job storage
-- Run this in your Supabase SQL editor

create table if not exists public.creator_projects (
  id text primary key,
  data jsonb not null,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create index if not exists idx_creator_projects_updated
  on public.creator_projects(updated_at desc);

create table if not exists public.creator_jobs (
  id text primary key,
  project_id text not null references public.creator_projects(id) on delete cascade,
  data jsonb not null,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create index if not exists idx_creator_jobs_project
  on public.creator_jobs(project_id);

-- Storage bucket for binary artifacts (figures, notebooks, narration audio)
-- Run via Supabase dashboard or API:
--   supabase storage create creator-artifacts --public=false --file-size-limit=100MB
