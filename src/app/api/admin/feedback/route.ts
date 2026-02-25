import { NextRequest, NextResponse } from 'next/server';
import { isAdmin } from '@/lib/admin';
import { getDb } from '@/lib/db';

export async function GET(req: NextRequest) {
  if (!(await isAdmin())) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 403 });
  }

  const { searchParams } = req.nextUrl;
  const page = parseInt(searchParams.get('page') || '1');
  const limit = 20;
  const offset = (page - 1) * limit;
  const type = searchParams.get('type');
  const courseSlug = searchParams.get('courseSlug');
  const dateFrom = searchParams.get('dateFrom');
  const dateTo = searchParams.get('dateTo');

  const db = getDb();

  // Build filtered query for feedback list
  let query = db
    .from('feedback')
    .select('*, users!inner(full_name, email)', { count: 'exact' })
    .order('created_at', { ascending: false })
    .range(offset, offset + limit - 1);

  if (type) query = query.eq('type', type);
  if (courseSlug) query = query.eq('course_slug', courseSlug);
  if (dateFrom) query = query.gte('created_at', dateFrom);
  if (dateTo) query = query.lte('created_at', dateTo + 'T23:59:59Z');

  const { data: rows, count, error } = await query;

  if (error) {
    return NextResponse.json({ error: 'Failed to fetch feedback' }, { status: 500 });
  }

  // Fetch tags for thumbs feedback
  const thumbsIds = (rows ?? []).filter(r => r.type === 'thumbs').map(r => r.id);
  let tagsMap: Record<string, string[]> = {};
  if (thumbsIds.length > 0) {
    const { data: tagRows } = await db
      .from('feedback_tags')
      .select('feedback_id, tag')
      .in('feedback_id', thumbsIds);
    for (const row of tagRows ?? []) {
      if (!tagsMap[row.feedback_id]) tagsMap[row.feedback_id] = [];
      tagsMap[row.feedback_id].push(row.tag);
    }
  }

  const feedback = (rows ?? []).map(r => ({
    id: r.id,
    userId: r.user_id,
    type: r.type,
    courseSlug: r.course_slug,
    podSlug: r.pod_slug,
    contentType: r.content_type,
    notebookOrder: r.notebook_order,
    rating: r.rating,
    comment: r.comment,
    surveyData: r.survey_data,
    category: r.category,
    pageUrl: r.page_url,
    createdAt: r.created_at,
    tags: tagsMap[r.id] || [],
    userName: r.users?.full_name,
    userEmail: r.users?.email,
  }));

  // Compute aggregate stats (unfiltered)
  const { data: allFeedback } = await db
    .from('feedback')
    .select('type, rating, created_at');

  const all = allFeedback ?? [];
  const now = new Date();
  const weekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000).toISOString();

  const emojiRatings = all.filter(r => r.type === 'emoji' && r.rating != null).map(r => r.rating as number);
  const npsRatings = all.filter(r => r.type === 'nps' && r.rating != null).map(r => r.rating as number);
  const thumbsRatings = all.filter(r => r.type === 'thumbs' && r.rating != null).map(r => r.rating as number);

  const avg = (arr: number[]) => arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : null;

  const npsDistribution: Record<number, number> = {};
  for (let i = 1; i <= 10; i++) npsDistribution[i] = 0;
  for (const r of npsRatings) npsDistribution[r] = (npsDistribution[r] || 0) + 1;

  // Tag breakdown
  const { data: allTags } = await db.from('feedback_tags').select('tag');
  const tagBreakdown: Record<string, number> = {};
  for (const t of allTags ?? []) {
    tagBreakdown[t.tag] = (tagBreakdown[t.tag] || 0) + 1;
  }

  const stats = {
    totalCount: all.length,
    avgNps: avg(npsRatings),
    avgEmoji: avg(emojiRatings),
    thumbsUpPercent: thumbsRatings.length > 0
      ? (thumbsRatings.filter(r => r === 1).length / thumbsRatings.length) * 100
      : null,
    last7DaysCount: all.filter(r => r.created_at >= weekAgo).length,
    npsDistribution,
    tagBreakdown,
  };

  return NextResponse.json({
    feedback,
    stats,
    total: count ?? 0,
    page,
    totalPages: Math.ceil((count ?? 0) / limit),
  });
}
