-- Feedback table: stores all feedback types (emoji, nps, thumbs, survey, general)
create table public.feedback (
  id text primary key,
  user_id text not null references public.users(id),
  type text not null check (type in ('emoji', 'nps', 'thumbs', 'survey', 'general')),
  course_slug text,
  pod_slug text,
  content_type text check (content_type in ('article', 'notebook', 'case-study', 'pod', 'course')),
  notebook_order integer,
  rating integer,
  comment text,
  survey_data jsonb,
  category text check (category in ('bug', 'suggestion', 'content', 'other')),
  page_url text,
  created_at timestamptz not null default now()
);

-- Prevent duplicate feedback per user per content item (for emoji/nps/thumbs)
create unique index feedback_unique_per_content
  on public.feedback (user_id, type, course_slug, pod_slug, content_type, notebook_order)
  where type in ('emoji', 'nps', 'thumbs');

-- Index for admin queries
create index feedback_created_at_idx on public.feedback (created_at desc);
create index feedback_type_idx on public.feedback (type);
create index feedback_course_idx on public.feedback (course_slug);

-- Feedback tags table: stores tags for thumbs feedback
create table public.feedback_tags (
  id text primary key,
  feedback_id text not null references public.feedback(id) on delete cascade,
  tag text not null check (tag in ('too_easy', 'too_hard', 'great_examples', 'needs_more_code', 'confusing'))
);

create index feedback_tags_feedback_id_idx on public.feedback_tags (feedback_id);

-- Enable RLS (service role key bypasses RLS, but good practice)
alter table public.feedback enable row level security;
alter table public.feedback_tags enable row level security;
