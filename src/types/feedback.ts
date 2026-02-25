export type FeedbackType = 'emoji' | 'nps' | 'thumbs' | 'survey' | 'general';

export type ContentType = 'article' | 'notebook' | 'case-study' | 'pod' | 'course';

export type FeedbackTag = 'too_easy' | 'too_hard' | 'great_examples' | 'needs_more_code' | 'confusing';

export type FeedbackCategory = 'bug' | 'suggestion' | 'content' | 'other';

export interface FeedbackSubmission {
  type: FeedbackType;
  courseSlug?: string;
  podSlug?: string;
  contentType?: ContentType;
  notebookOrder?: number;
  rating?: number;
  comment?: string;
  tags?: FeedbackTag[];
  surveyData?: Record<string, string | number>;
  category?: FeedbackCategory;
  pageUrl?: string;
}

export interface FeedbackRecord {
  id: string;
  userId: string;
  type: FeedbackType;
  courseSlug: string | null;
  podSlug: string | null;
  contentType: ContentType | null;
  notebookOrder: number | null;
  rating: number | null;
  comment: string | null;
  surveyData: Record<string, string | number> | null;
  category: FeedbackCategory | null;
  pageUrl: string | null;
  createdAt: string;
  tags?: FeedbackTag[];
  userName?: string;
  userEmail?: string;
}

export interface FeedbackStats {
  totalCount: number;
  avgNps: number | null;
  avgEmoji: number | null;
  thumbsUpPercent: number | null;
  last7DaysCount: number;
  npsDistribution: Record<number, number>;
  tagBreakdown: Record<string, number>;
}
