import { BrevoClient } from '@getbrevo/brevo';

interface SendReplyEmailParams {
  to: string;
  toName: string;
  subject: string;
  html: string;
}

export async function sendReplyEmail({ to, toName, subject, html }: SendReplyEmailParams) {
  const apiKey = process.env.BREVO_API_KEY;
  if (!apiKey) {
    throw new Error('Missing BREVO_API_KEY environment variable');
  }

  const client = new BrevoClient({ apiKey });

  const result = await client.transactionalEmails.sendTransacEmail({
    subject,
    htmlContent: html,
    sender: { name: 'Vizuara', email: 'hello@vizuara.com' },
    to: [{ email: to, name: toName }],
  });

  return result;
}
