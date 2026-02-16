import type { CourseProgress, CertificateData } from '@/types/course';

const STORAGE_KEY_PREFIX = 'vizuara_progress_';
const CERT_KEY_PREFIX = 'vizuara_cert_';

const DEFAULTS: CourseProgress = {
  articleRead: false,
  completedNotebooks: [],
  caseStudyComplete: false,
  lastVisited: '',
};

// --- User scoping ---
// Progress is scoped per user. The current user ID is persisted in localStorage
// so it survives dev server hot reloads (module-level variables reset on HMR).
const USER_KEY = 'vizuara_current_user';

function getCurrentUserId(): string {
  if (typeof window === 'undefined') return 'anon';
  return localStorage.getItem(USER_KEY) || 'anon';
}

/**
 * Migrate progress from one key namespace to another, merging if the target already has data.
 * Handles both old unscoped keys (no userId) and anon-scoped keys.
 */
function migrateKeys(sourcePrefix: string, targetUserId: string): void {
  if (typeof window === 'undefined') return;

  for (const basePrefix of [STORAGE_KEY_PREFIX, CERT_KEY_PREFIX]) {
    const fullSource = `${basePrefix}${sourcePrefix}`;
    const keysToProcess: { oldKey: string; slug: string }[] = [];

    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key && key.startsWith(fullSource)) {
        const slug = key.slice(fullSource.length);
        keysToProcess.push({ oldKey: key, slug });
      }
    }

    for (const { oldKey, slug } of keysToProcess) {
      const newKey = `${basePrefix}${targetUserId}_${slug}`;
      const sourceData = localStorage.getItem(oldKey);
      if (!sourceData) continue;

      const existingData = localStorage.getItem(newKey);
      if (existingData && basePrefix === STORAGE_KEY_PREFIX) {
        // Merge progress: combine both sets
        try {
          const src = { ...DEFAULTS, ...JSON.parse(sourceData) } as CourseProgress;
          const dst = { ...DEFAULTS, ...JSON.parse(existingData) } as CourseProgress;
          const merged: CourseProgress = {
            articleRead: dst.articleRead || src.articleRead,
            completedNotebooks: [...new Set([...dst.completedNotebooks, ...src.completedNotebooks])],
            caseStudyComplete: dst.caseStudyComplete || src.caseStudyComplete,
            lastVisited: dst.lastVisited > src.lastVisited ? dst.lastVisited : src.lastVisited,
          };
          localStorage.setItem(newKey, JSON.stringify(merged));
        } catch {
          // If parsing fails, keep existing
        }
      } else if (!existingData) {
        localStorage.setItem(newKey, sourceData);
      }
      localStorage.removeItem(oldKey);
    }
  }
}

/** Call this when auth state changes (login, signup, logout, initial load). */
export function setProgressUser(userId: string | null): void {
  if (typeof window === 'undefined') return;
  const newId = userId || 'anon';
  const previousId = getCurrentUserId();

  if (newId !== previousId) {
    localStorage.setItem(USER_KEY, newId);

    if (newId !== 'anon') {
      // Migrate old unscoped keys (vizuara_progress_<slug> with no underscore in slug)
      const migrationFlag = `vizuara_migrated_${newId}`;
      if (!localStorage.getItem(migrationFlag)) {
        for (const prefix of [STORAGE_KEY_PREFIX, CERT_KEY_PREFIX]) {
          const oldKeys: { key: string; slug: string }[] = [];
          for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i);
            if (key && key.startsWith(prefix)) {
              const rest = key.slice(prefix.length);
              if (!rest.includes('_')) {
                oldKeys.push({ key, slug: rest });
              }
            }
          }
          for (const { key, slug } of oldKeys) {
            const newKey = `${prefix}${newId}_${slug}`;
            if (!localStorage.getItem(newKey)) {
              const data = localStorage.getItem(key);
              if (data) localStorage.setItem(newKey, data);
            }
            localStorage.removeItem(key);
          }
        }
        localStorage.setItem(migrationFlag, '1');
      }

      // Migrate anon keys to the real user (handles race condition
      // where progress was written before auth resolved on page load)
      if (previousId === 'anon') {
        migrateKeys('anon_', newId);
      }
    }

    notifyListeners();
  }
}

// --- Event emitter for reactive progress updates ---
type ProgressListener = () => void;
const listeners = new Set<ProgressListener>();

export function onProgressChange(listener: ProgressListener): () => void {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
}

function notifyListeners() {
  listeners.forEach((l) => l());
}

// --- Core functions ---

function getStorageKey(courseSlug: string): string {
  return `${STORAGE_KEY_PREFIX}${getCurrentUserId()}_${courseSlug}`;
}

export function getProgress(courseSlug: string): CourseProgress {
  if (typeof window === 'undefined') return { ...DEFAULTS };
  try {
    const raw = localStorage.getItem(getStorageKey(courseSlug));
    if (!raw) return { ...DEFAULTS };
    return { ...DEFAULTS, ...JSON.parse(raw) };
  } catch {
    return { ...DEFAULTS };
  }
}

export function setProgress(courseSlug: string, progress: CourseProgress): void {
  if (typeof window === 'undefined') return;
  localStorage.setItem(getStorageKey(courseSlug), JSON.stringify(progress));
  notifyListeners();
}

export function clearAllProgress(): void {
  if (typeof window === 'undefined') return;
  const uid = getCurrentUserId();
  const userPrefix = `${STORAGE_KEY_PREFIX}${uid}_`;
  const certPrefix = `${CERT_KEY_PREFIX}${uid}_`;
  const keysToRemove: string[] = [];
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (key && (key.startsWith(userPrefix) || key.startsWith(certPrefix))) {
      keysToRemove.push(key);
    }
  }
  keysToRemove.forEach((key) => localStorage.removeItem(key));
  notifyListeners();
}

export function markArticleRead(courseSlug: string): void {
  const progress = getProgress(courseSlug);
  progress.articleRead = true;
  progress.lastVisited = new Date().toISOString();
  setProgress(courseSlug, progress);
}

export function markNotebookComplete(courseSlug: string, notebookSlug: string): void {
  const progress = getProgress(courseSlug);
  if (!progress.completedNotebooks.includes(notebookSlug)) {
    progress.completedNotebooks.push(notebookSlug);
  }
  progress.lastVisited = new Date().toISOString();
  setProgress(courseSlug, progress);
}

export function isNotebookComplete(courseSlug: string, notebookSlug: string): boolean {
  const progress = getProgress(courseSlug);
  return progress.completedNotebooks.includes(notebookSlug);
}

export function markCaseStudyComplete(courseSlug: string): void {
  const progress = getProgress(courseSlug);
  progress.caseStudyComplete = true;
  progress.lastVisited = new Date().toISOString();
  setProgress(courseSlug, progress);
}

export function isCaseStudyComplete(courseSlug: string): boolean {
  return getProgress(courseSlug).caseStudyComplete;
}

export function isCourseFullyComplete(
  courseSlug: string,
  totalNotebooks: number,
  hasCaseStudy: boolean
): boolean {
  const p = getProgress(courseSlug);
  if (!p.articleRead) return false;
  if (totalNotebooks > 0 && p.completedNotebooks.length < totalNotebooks) return false;
  if (hasCaseStudy && !p.caseStudyComplete) return false;
  return true;
}

export function getCourseCompletion(
  courseSlug: string,
  totalNotebooks: number,
  hasCaseStudy: boolean = false
): number {
  const progress = getProgress(courseSlug);
  const caseStudyWeight = hasCaseStudy ? 1 : 0;
  const totalSteps = 1 + totalNotebooks + caseStudyWeight;
  const completedSteps =
    (progress.articleRead ? 1 : 0) +
    progress.completedNotebooks.length +
    (hasCaseStudy && progress.caseStudyComplete ? 1 : 0);
  return Math.round((completedSteps / totalSteps) * 100);
}

// --- Certificate ---

function generateCertificateId(courseSlug: string): string {
  const timestamp = Date.now().toString(16).toUpperCase();
  const slug = courseSlug.replace(/[^a-zA-Z0-9]/g, '').substring(0, 8).toUpperCase();
  return `VIZ-${slug}-${timestamp}`;
}

function getCertKey(courseSlug: string): string {
  return `${CERT_KEY_PREFIX}${getCurrentUserId()}_${courseSlug}`;
}

export function getCertificate(courseSlug: string): CertificateData | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = localStorage.getItem(getCertKey(courseSlug));
    return raw ? (JSON.parse(raw) as CertificateData) : null;
  } catch {
    return null;
  }
}

export function issueCertificate(
  courseSlug: string,
  courseTitle: string,
  difficulty: 'beginner' | 'intermediate' | 'advanced',
  estimatedHours: number,
  notebookCount: number,
  studentName: string = 'Student'
): CertificateData {
  const existing = getCertificate(courseSlug);
  if (existing) return existing;

  const cert: CertificateData = {
    certificateId: generateCertificateId(courseSlug),
    studentName,
    courseTitle,
    courseSlug,
    completionDate: new Date().toISOString(),
    difficulty,
    estimatedHours,
    notebookCount,
  };
  if (typeof window !== 'undefined') {
    localStorage.setItem(getCertKey(courseSlug), JSON.stringify(cert));
  }
  return cert;
}
