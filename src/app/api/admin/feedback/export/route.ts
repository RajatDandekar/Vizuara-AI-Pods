import { NextRequest, NextResponse } from 'next/server';
import { isAdmin } from '@/lib/admin';
import { getDb } from '@/lib/db';

export async function GET(req: NextRequest) {
  if (!(await isAdmin())) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 403 });
  }

  const { searchParams } = req.nextUrl;
  const type = searchParams.get('type');
  const courseSlug = searchParams.get('courseSlug');
  const dateFrom = searchParams.get('dateFrom');
  const dateTo = searchParams.get('dateTo');

  const db = getDb();

  let query = db
    .from('feedback')
    .select('*, users!inner(full_name, email)')
    .order('created_at', { ascending: false });

  if (type) query = query.eq('type', type);
  if (courseSlug) query = query.eq('course_slug', courseSlug);
  if (dateFrom) query = query.gte('created_at', dateFrom);
  if (dateTo) query = query.lte('created_at', dateTo + 'T23:59:59Z');

  const { data: rows, error } = await query;

  if (error) {
    return NextResponse.json({ error: 'Failed to export' }, { status: 500 });
  }

  // Fetch all tags
  const allIds = (rows ?? []).map(r => r.id);
  let tagsMap: Record<string, string[]> = {};
  if (allIds.length > 0) {
    // Batch in groups of 100
    for (let i = 0; i < allIds.length; i += 100) {
      const batch = allIds.slice(i, i + 100);
      const { data: tagRows } = await db
        .from('feedback_tags')
        .select('feedback_id, tag')
        .in('feedback_id', batch);
      for (const row of tagRows ?? []) {
        if (!tagsMap[row.feedback_id]) tagsMap[row.feedback_id] = [];
        tagsMap[row.feedback_id].push(row.tag);
      }
    }
  }

  // Build CSV
  const headers = ['Date', 'User', 'Email', 'Type', 'Course', 'Pod', 'Content Type', 'Notebook #', 'Rating', 'Comment', 'Tags', 'Category', 'Survey Data', 'Page URL'];
  const csvRows = [headers.join(',')];

  for (const r of rows ?? []) {
    const row = [
      r.created_at,
      `"${(r.users?.full_name || '').replace(/"/g, '""')}"`,
      r.users?.email || '',
      r.type,
      r.course_slug || '',
      r.pod_slug || '',
      r.content_type || '',
      r.notebook_order ?? '',
      r.rating ?? '',
      `"${(r.comment || '').replace(/"/g, '""')}"`,
      (tagsMap[r.id] || []).join(';'),
      r.category || '',
      r.survey_data ? `"${JSON.stringify(r.survey_data).replace(/"/g, '""')}"` : '',
      r.page_url || '',
    ];
    csvRows.push(row.join(','));
  }

  return new NextResponse(csvRows.join('\n'), {
    headers: {
      'Content-Type': 'text/csv',
      'Content-Disposition': `attachment; filename="feedback-export-${new Date().toISOString().split('T')[0]}.csv"`,
    },
  });
}
