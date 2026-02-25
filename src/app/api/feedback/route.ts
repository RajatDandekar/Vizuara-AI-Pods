import { NextRequest, NextResponse } from 'next/server';
import { getAuthUser, generateId } from '@/lib/auth';
import { getDb } from '@/lib/db';
import type { FeedbackSubmission } from '@/types/feedback';

export async function POST(req: NextRequest) {
  const user = await getAuthUser();
  if (!user) {
    return NextResponse.json({ error: 'Not authenticated' }, { status: 401 });
  }

  const body: FeedbackSubmission = await req.json();

  if (!body.type) {
    return NextResponse.json({ error: 'Missing feedback type' }, { status: 400 });
  }

  const db = getDb();
  const feedbackId = generateId();

  // For emoji/nps/thumbs: upsert to prevent duplicates
  if (['emoji', 'nps', 'thumbs'].includes(body.type)) {
    // Check for existing feedback
    let query = db
      .from('feedback')
      .select('id')
      .eq('user_id', user.id)
      .eq('type', body.type);

    if (body.courseSlug) query = query.eq('course_slug', body.courseSlug);
    else query = query.is('course_slug', null);

    if (body.podSlug) query = query.eq('pod_slug', body.podSlug);
    else query = query.is('pod_slug', null);

    if (body.contentType) query = query.eq('content_type', body.contentType);
    else query = query.is('content_type', null);

    if (body.notebookOrder !== undefined) query = query.eq('notebook_order', body.notebookOrder);
    else query = query.is('notebook_order', null);

    const { data: existing } = await query.maybeSingle();

    if (existing) {
      // Update existing
      const { error: updateError } = await db
        .from('feedback')
        .update({
          rating: body.rating,
          comment: body.comment,
        })
        .eq('id', existing.id);

      if (updateError) {
        return NextResponse.json({ error: 'Failed to update feedback' }, { status: 500 });
      }

      // Update tags if thumbs
      if (body.type === 'thumbs' && body.tags) {
        await db.from('feedback_tags').delete().eq('feedback_id', existing.id);
        if (body.tags.length > 0) {
          await db.from('feedback_tags').insert(
            body.tags.map(tag => ({ id: generateId(), feedback_id: existing.id, tag }))
          );
        }
      }

      return NextResponse.json({ id: existing.id, updated: true });
    }
  }

  // Insert new feedback
  const { error: insertError } = await db.from('feedback').insert({
    id: feedbackId,
    user_id: user.id,
    type: body.type,
    course_slug: body.courseSlug || null,
    pod_slug: body.podSlug || null,
    content_type: body.contentType || null,
    notebook_order: body.notebookOrder ?? null,
    rating: body.rating ?? null,
    comment: body.comment || null,
    survey_data: body.surveyData || null,
    category: body.category || null,
    page_url: body.pageUrl || null,
  });

  if (insertError) {
    return NextResponse.json({ error: 'Failed to submit feedback' }, { status: 500 });
  }

  // Insert tags for thumbs feedback
  if (body.type === 'thumbs' && body.tags && body.tags.length > 0) {
    await db.from('feedback_tags').insert(
      body.tags.map(tag => ({ id: generateId(), feedback_id: feedbackId, tag }))
    );
  }

  return NextResponse.json({ id: feedbackId, updated: false });
}

export async function GET(req: NextRequest) {
  const user = await getAuthUser();
  if (!user) {
    return NextResponse.json({ error: 'Not authenticated' }, { status: 401 });
  }

  const { searchParams } = req.nextUrl;
  const type = searchParams.get('type');

  if (!type) {
    return NextResponse.json({ error: 'Missing type parameter' }, { status: 400 });
  }

  const courseSlug = searchParams.get('courseSlug');
  const podSlug = searchParams.get('podSlug');
  const contentType = searchParams.get('contentType');
  const notebookOrder = searchParams.get('notebookOrder');

  const db = getDb();
  let query = db
    .from('feedback')
    .select('*')
    .eq('user_id', user.id)
    .eq('type', type);

  if (courseSlug) query = query.eq('course_slug', courseSlug);
  else query = query.is('course_slug', null);

  if (podSlug) query = query.eq('pod_slug', podSlug);
  else query = query.is('pod_slug', null);

  if (contentType) query = query.eq('content_type', contentType);
  else query = query.is('content_type', null);

  if (notebookOrder) query = query.eq('notebook_order', parseInt(notebookOrder));
  else query = query.is('notebook_order', null);

  const { data, error } = await query.maybeSingle();

  if (error) {
    return NextResponse.json({ error: 'Failed to fetch feedback' }, { status: 500 });
  }

  if (!data) {
    return NextResponse.json({ feedback: null });
  }

  // Fetch tags if thumbs
  let tags: string[] = [];
  if (data.type === 'thumbs') {
    const { data: tagRows } = await db
      .from('feedback_tags')
      .select('tag')
      .eq('feedback_id', data.id);
    tags = (tagRows ?? []).map((r: { tag: string }) => r.tag);
  }

  return NextResponse.json({
    feedback: {
      id: data.id,
      userId: data.user_id,
      type: data.type,
      courseSlug: data.course_slug,
      podSlug: data.pod_slug,
      contentType: data.content_type,
      notebookOrder: data.notebook_order,
      rating: data.rating,
      comment: data.comment,
      surveyData: data.survey_data,
      category: data.category,
      pageUrl: data.page_url,
      createdAt: data.created_at,
      tags,
    },
  });
}
