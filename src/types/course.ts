// Course platform types — Course → Pod hierarchy

// ─── Shared / unchanged ──────────────────────────────────────────────

export interface NotebookMeta {
  title: string;
  slug: string;
  objective: string;
  colabUrl: string;
  downloadPath?: string;
  hasNarration?: boolean;
  estimatedMinutes: number;
  todoCount: number;
  order: number;
}

export interface CaseStudyMeta {
  title: string;
  subtitle: string;
  company: string;
  industry: string;
  description: string;
  pdfPath: string;
  colabUrl: string;
  notebookPath: string;
}

export interface CuratorInfo {
  name: string;
  title?: string;
  bio?: string;
  videoUrl?: string;
  imageUrl?: string;
}

export interface CaseStudySection {
  id: string;
  title: string;
  content: string;
}

// ─── Pod types (a pod = what used to be a "course") ──────────────────

/** Lightweight listing card for a pod within a course */
export interface PodCard {
  slug: string;
  title: string;
  description: string;
  order: number;
  notebookCount: number;
  estimatedHours: number;
  hasCaseStudy: boolean;
  thumbnail?: string;
}

/** Full pod definition — the manifest that lives in pod.json */
export interface PodManifest {
  title: string;
  slug: string;
  courseSlug: string;
  order: number;
  description: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  estimatedHours: number;
  thumbnail?: string;
  prerequisites: string[];
  tags: string[];
  article: {
    notionUrl?: string;
    figureUrls: Record<string, string>;
  };
  notebooks: NotebookMeta[];
  caseStudy?: CaseStudyMeta;
  curator?: CuratorInfo;
}

/** Progress for a single pod (same shape as old CourseProgress) */
export interface PodProgress {
  articleRead: boolean;
  completedNotebooks: string[];
  caseStudyComplete: boolean;
  lastVisited: string;
}

// ─── Course types (a course contains multiple pods) ──────────────────

/** Course-level card shown on homepage / catalog */
export interface CourseCard {
  slug: string;
  title: string;
  description: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  estimatedHours: number;
  thumbnail?: string;
  tags: string[];
  podCount: number;
  totalNotebooks: number;
  status?: 'live' | 'upcoming' | 'archived' | 'draft';
  expectedLaunchDate?: string;
  /** @deprecated — use podCount/totalNotebooks instead */
  notebookCount?: number;
}

/** Full course manifest — lives in course.json */
export interface CourseManifest {
  title: string;
  slug: string;
  description: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  estimatedHours: number;
  thumbnail?: string;
  tags: string[];
  pods: PodCard[];
}

/** Aggregated course-level progress */
export interface CourseProgressSummary {
  completedPods: number;
  totalPods: number;
  percentage: number;
}

// ─── Free pod showcase (homepage guest view) ─────────────────────────

export interface FreePodShowcase {
  courseSlug: string;
  courseTitle: string;
  podSlug: string;
  title: string;
  description: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  estimatedHours: number;
  notebooks: NotebookMeta[];
  caseStudy?: CaseStudyMeta;
  thumbnail?: string;
  tags: string[];
}

// ─── Certificate types ───────────────────────────────────────────────

/** Certificate for completing a single pod */
export interface CertificateData {
  certificateId: string;
  studentName: string;
  courseTitle: string;
  courseSlug: string;
  podSlug?: string;
  completionDate: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  estimatedHours: number;
  notebookCount: number;
}

/** Certificate for completing all pods in a course */
export interface CourseCertificateData {
  certificateId: string;
  studentName: string;
  courseTitle: string;
  courseSlug: string;
  completionDate: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  estimatedHours: number;
  podCount: number;
  totalNotebooks: number;
}
