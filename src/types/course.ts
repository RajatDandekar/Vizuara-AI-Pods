// Course platform types

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

export interface CourseManifest {
  title: string;
  slug: string;
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

export interface CourseCard {
  slug: string;
  title: string;
  description: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  estimatedHours: number;
  thumbnail?: string;
  tags: string[];
  notebookCount: number;
  status?: 'live' | 'upcoming' | 'archived' | 'draft';
  expectedLaunchDate?: string;
}

export interface CourseProgress {
  articleRead: boolean;
  completedNotebooks: string[];
  caseStudyComplete: boolean;
  lastVisited: string;
}

export interface CertificateData {
  certificateId: string;
  studentName: string;
  courseTitle: string;
  courseSlug: string;
  completionDate: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  estimatedHours: number;
  notebookCount: number;
}
