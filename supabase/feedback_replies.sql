create table public.feedback_replies (
  id text primary key,
  feedback_id text not null references public.feedback(id) on delete cascade,
  reply_text text not null,
  replied_by text not null,
  status text not null default 'draft' check (status in ('draft', 'sent')),
  sent_at timestamptz,
  created_at timestamptz not null default now()
);

create index feedback_replies_feedback_id_idx on public.feedback_replies (feedback_id);

alter table public.feedback_replies enable row level security;
