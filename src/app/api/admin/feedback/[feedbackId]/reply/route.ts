import { NextRequest, NextResponse } from 'next/server';
import { isAdmin, getAdminUser } from '@/lib/admin';
import { getDb } from '@/lib/db';
import { sendReplyEmail } from '@/lib/email';

export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ feedbackId: string }> }
) {
  if (!(await isAdmin())) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 403 });
  }

  const { feedbackId } = await params;
  const { replyText } = await req.json();

  if (!replyText || typeof replyText !== 'string') {
    return NextResponse.json({ error: 'replyText is required' }, { status: 400 });
  }

  const admin = await getAdminUser();
  const db = getDb();

  const id = `reply_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

  const { data, error } = await db
    .from('feedback_replies')
    .insert({
      id,
      feedback_id: feedbackId,
      reply_text: replyText,
      replied_by: admin?.fullName || 'Admin',
      status: 'draft',
    })
    .select()
    .single();

  if (error) {
    return NextResponse.json({ error: 'Failed to save reply' }, { status: 500 });
  }

  return NextResponse.json({
    id: data.id,
    feedbackId: data.feedback_id,
    replyText: data.reply_text,
    repliedBy: data.replied_by,
    status: data.status,
    sentAt: data.sent_at,
    createdAt: data.created_at,
  });
}

export async function PUT(
  req: NextRequest,
  { params }: { params: Promise<{ feedbackId: string }> }
) {
  if (!(await isAdmin())) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 403 });
  }

  const { feedbackId } = await params;
  const { replyId } = await req.json();

  if (!replyId) {
    return NextResponse.json({ error: 'replyId is required' }, { status: 400 });
  }

  const db = getDb();

  // Fetch the reply
  const { data: reply, error: replyErr } = await db
    .from('feedback_replies')
    .select('*')
    .eq('id', replyId)
    .eq('feedback_id', feedbackId)
    .single();

  if (replyErr || !reply) {
    return NextResponse.json({ error: 'Reply not found' }, { status: 404 });
  }

  // Fetch the feedback record with user info
  const { data: feedback, error: fbErr } = await db
    .from('feedback')
    .select('*, users!inner(full_name, email)')
    .eq('id', feedbackId)
    .single();

  if (fbErr || !feedback) {
    return NextResponse.json({ error: 'Feedback not found' }, { status: 404 });
  }

  const userEmail = feedback.users?.email;
  const userName = feedback.users?.full_name || 'Student';

  if (!userEmail) {
    return NextResponse.json({ error: 'No email found for this user' }, { status: 400 });
  }

  // Build the email
  const subject = `Re: Your feedback on Vizuara`;
  const html = `
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 600px; margin: 0 auto;">
      <p>Hi ${userName},</p>
      <p>Thank you for your feedback on Vizuara. Here's our response:</p>
      <div style="background: #f8f9fa; border-left: 4px solid #4f46e5; padding: 16px; margin: 16px 0; border-radius: 4px;">
        ${reply.reply_text.replace(/\n/g, '<br>')}
      </div>
      <p>If you have any more questions, feel free to reply to this email.</p>
      <p>Best,<br>The Vizuara Team</p>
    </div>
  `;

  // Send via Brevo
  try {
    await sendReplyEmail({ to: userEmail, toName: userName, subject, html });
  } catch (err) {
    console.error('Failed to send reply email:', err);
    return NextResponse.json({ error: 'Failed to send email' }, { status: 500 });
  }

  // Update reply status
  const { error: updateErr } = await db
    .from('feedback_replies')
    .update({ status: 'sent', sent_at: new Date().toISOString() })
    .eq('id', replyId);

  if (updateErr) {
    return NextResponse.json({ error: 'Email sent but failed to update status' }, { status: 500 });
  }

  return NextResponse.json({ success: true });
}
