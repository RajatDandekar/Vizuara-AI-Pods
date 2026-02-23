import { NextResponse } from 'next/server';
import { getAuthUser } from '@/lib/auth';
import { getAdminFirestore } from '@/lib/firebase-admin';

export const dynamic = 'force-dynamic';

const PODS_ENROLLMENT_ID = process.env.PODS_ENROLLMENT_ID || 'course_20006198';

export async function GET() {
  try {
    const user = await getAuthUser();
    if (!user) {
      return NextResponse.json({ enrolled: false, status: null, enrollment: null });
    }

    const db = getAdminFirestore();
    const docRef = db.collection('Enrollments').doc(`${user.id}_${PODS_ENROLLMENT_ID}`);
    const doc = await docRef.get();

    if (!doc.exists) {
      return NextResponse.json({ enrolled: false, status: null, enrollment: null });
    }

    const data = doc.data()!;
    const isActive = data.status === 'ACTIVE' || data.status === 'COMPLETED';

    return NextResponse.json({
      enrolled: isActive,
      status: data.status,
      enrollment: {
        uid: user.id,
        courseId: PODS_ENROLLMENT_ID,
        status: data.status,
        enrolledAt: data.enrollmentDate?.toDate?.()?.toISOString?.() ?? data.enrollmentDate ?? null,
      },
    });
  } catch {
    return NextResponse.json({ enrolled: false, status: null, enrollment: null });
  }
}
