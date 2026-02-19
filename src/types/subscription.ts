export type EnrollmentStatus = 'ACTIVE' | 'COMPLETED' | 'DROPPED';

export interface Enrollment {
  uid: string;
  courseId: string;
  status: EnrollmentStatus;
  enrolledAt?: string;
}

export interface EnrollmentStatusResponse {
  enrolled: boolean;
  status: EnrollmentStatus | null;
  enrollment: Enrollment | null;
}
