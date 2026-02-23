import { NextResponse } from 'next/server';
import { getAuthUser } from '@/lib/auth';
import { getAdminFirestore } from '@/lib/firebase-admin';

export const dynamic = 'force-dynamic';

const PODS_ENROLLMENT_ID = process.env.PODS_ENROLLMENT_ID || 'course_20006198';

export async function GET() {
  try {
    const user = await getAuthUser();
    if (!user) {
      return NextResponse.json({ enrolled: false, status: null, enrollment: null, debug: { reason: 'no_user' } });
    }

    const db = getAdminFirestore();
    const docId = `${user.id}_${PODS_ENROLLMENT_ID}`;
    const docRef = db.collection('Enrollments').doc(docId);
    const doc = await docRef.get();

    if (!doc.exists) {
      // Also check if any enrollment docs exist for this user at all
      const allEnrollments = await db.collection('Enrollments')
        .where('userId', '==', user.id)
        .limit(5)
        .get();
      const altDocs = allEnrollments.docs.map((d) => ({ id: d.id, ...d.data() }));

      return NextResponse.json({
        enrolled: false, status: null, enrollment: null,
        debug: {
          reason: 'doc_not_found',
          lookingFor: docId,
          uid: user.id,
          courseId: PODS_ENROLLMENT_ID,
          otherEnrollments: altDocs,
        },
      });
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
      debug: {
        docId,
        docData: data,
      },
    });
  } catch (err) {
    return NextResponse.json({
      enrolled: false, status: null, enrollment: null,
      debug: { reason: 'error', message: String(err) },
    });
  }
}
