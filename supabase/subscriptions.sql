-- Subscriptions table for Razorpay integration
-- Run this in Supabase SQL Editor

create table if not exists public.subscriptions (
  id uuid primary key default gen_random_uuid(),
  user_id text not null references public.users(id) on delete cascade,
  razorpay_subscription_id text unique,
  razorpay_plan_id text,
  status text not null default 'created',
  current_start timestamptz,
  current_end timestamptz,
  last_payment_id text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- Index for fast lookups by user
create index if not exists idx_subscriptions_user_id on public.subscriptions(user_id);

-- Index for webhook lookups by razorpay subscription id
create index if not exists idx_subscriptions_razorpay_id on public.subscriptions(razorpay_subscription_id);

-- RLS policies (service role bypasses these, but good practice)
alter table public.subscriptions enable row level security;
