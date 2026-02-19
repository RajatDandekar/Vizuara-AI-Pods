import type { PodProgress, CertificateData, CourseCertificateData, PodCard } from '@/types/course';

const STORAGE_KEY_PREFIX = 'vizuara_progress_';
const CERT_KEY_PREFIX = 'vizuara_cert_';
const COURSE_CERT_KEY_PREFIX = 'vizuara_coursecert_';

const DEFAULTS: PodProgress = {
  articleRead: false,
  completedNotebooks: [],
  caseStudyComplete: false,
  lastVisited: '',
};

// --- User scoping ---
const USER_KEY = 'vizuara_current_user';

function getCurrentUserId(): string {
  if (typeof window === 'undefined') return 'anon';
  return localStorage.getItem(USER_KEY) || 'anon';
}

/**
 * Migrate progress from one key namespace to another, merging if the target already has data.
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
        try {
          const src = { ...DEFAULTS, ...JSON.parse(sourceData) } as PodProgress;
          const dst = { ...DEFAULTS, ...JSON.parse(existingData) } as PodProgress;
          const merged: PodProgress = {
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

      if (previousId === 'anon') {
        migrateKeys('anon_', newId);
      }
    }

    notifyListeners();
  }
}

// --- Event emitter ---
type ProgressListener = () => void;
const listeners = new Set<ProgressListener>();

export function onProgressChange(listener: ProgressListener): () => void {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
}

function notifyListeners() {
  listeners.forEach((l) => l());
}

// ─── Pod-level progress ──────────────────────────────────────────────

/** Storage key for pod progress. Uses double underscore to separate course and pod slugs. */
function getPodStorageKey(courseSlug: string, podSlug: string): string {
  return `${STORAGE_KEY_PREFIX}${getCurrentUserId()}_${courseSlug}__${podSlug}`;
}

/** Legacy storage key (old flat structure — single slug). */
function getLegacyStorageKey(slug: string): string {
  return `${STORAGE_KEY_PREFIX}${getCurrentUserId()}_${slug}`;
}

export function getPodProgress(courseSlug: string, podSlug: string): PodProgress {
  if (typeof window === 'undefined') return { ...DEFAULTS };
  try {
    // Try new key first
    let raw = localStorage.getItem(getPodStorageKey(courseSlug, podSlug));
    // Fall back to legacy key (old flat structure)
    if (!raw) {
      raw = localStorage.getItem(getLegacyStorageKey(podSlug));
    }
    if (!raw) return { ...DEFAULTS };
    return { ...DEFAULTS, ...JSON.parse(raw) };
  } catch {
    return { ...DEFAULTS };
  }
}

export function setPodProgress(courseSlug: string, podSlug: string, progress: PodProgress): void {
  if (typeof window === 'undefined') return;
  localStorage.setItem(getPodStorageKey(courseSlug, podSlug), JSON.stringify(progress));
  notifyListeners();
}

export function markPodArticleRead(courseSlug: string, podSlug: string): void {
  const progress = getPodProgress(courseSlug, podSlug);
  progress.articleRead = true;
  progress.lastVisited = new Date().toISOString();
  setPodProgress(courseSlug, podSlug, progress);
}

export function markPodNotebookComplete(courseSlug: string, podSlug: string, notebookSlug: string): void {
  const progress = getPodProgress(courseSlug, podSlug);
  if (!progress.completedNotebooks.includes(notebookSlug)) {
    progress.completedNotebooks.push(notebookSlug);
  }
  progress.lastVisited = new Date().toISOString();
  setPodProgress(courseSlug, podSlug, progress);
}

export function isPodNotebookComplete(courseSlug: string, podSlug: string, notebookSlug: string): boolean {
  return getPodProgress(courseSlug, podSlug).completedNotebooks.includes(notebookSlug);
}

export function markPodCaseStudyComplete(courseSlug: string, podSlug: string): void {
  const progress = getPodProgress(courseSlug, podSlug);
  progress.caseStudyComplete = true;
  progress.lastVisited = new Date().toISOString();
  setPodProgress(courseSlug, podSlug, progress);
}

export function isPodCaseStudyComplete(courseSlug: string, podSlug: string): boolean {
  return getPodProgress(courseSlug, podSlug).caseStudyComplete;
}

export function isPodFullyComplete(
  courseSlug: string,
  podSlug: string,
  totalNotebooks: number,
  hasCaseStudy: boolean
): boolean {
  const p = getPodProgress(courseSlug, podSlug);
  if (!p.articleRead) return false;
  if (totalNotebooks > 0 && p.completedNotebooks.length < totalNotebooks) return false;
  if (hasCaseStudy && !p.caseStudyComplete) return false;
  return true;
}

export function getPodCompletion(
  courseSlug: string,
  podSlug: string,
  totalNotebooks: number,
  hasCaseStudy: boolean = false
): number {
  const progress = getPodProgress(courseSlug, podSlug);
  const caseStudyWeight = hasCaseStudy ? 1 : 0;
  const totalSteps = 1 + totalNotebooks + caseStudyWeight;
  if (totalSteps === 0) return 0;
  const completedSteps =
    (progress.articleRead ? 1 : 0) +
    progress.completedNotebooks.length +
    (hasCaseStudy && progress.caseStudyComplete ? 1 : 0);
  return Math.round((completedSteps / totalSteps) * 100);
}

// ─── Course-level aggregation ────────────────────────────────────────

export function getCourseCompletion(
  courseSlug: string,
  pods: PodCard[]
): { completedPods: number; totalPods: number; percentage: number } {
  let completedPods = 0;

  for (const pod of pods) {
    if (isPodFullyComplete(courseSlug, pod.slug, pod.notebookCount, pod.hasCaseStudy)) {
      completedPods++;
    }
  }

  const totalPods = pods.length;
  const percentage = totalPods > 0 ? Math.round((completedPods / totalPods) * 100) : 0;

  return { completedPods, totalPods, percentage };
}

export function isCourseComplete(courseSlug: string, pods: PodCard[]): boolean {
  return pods.every((pod) =>
    isPodFullyComplete(courseSlug, pod.slug, pod.notebookCount, pod.hasCaseStudy)
  );
}

// ─── Pod Certificate ─────────────────────────────────────────────────

function generateCertificateId(slug: string): string {
  const timestamp = Date.now().toString(16).toUpperCase();
  const cleanSlug = slug.replace(/[^a-zA-Z0-9]/g, '').substring(0, 8).toUpperCase();
  return `VIZ-${cleanSlug}-${timestamp}`;
}

function getPodCertKey(courseSlug: string, podSlug: string): string {
  return `${CERT_KEY_PREFIX}${getCurrentUserId()}_${courseSlug}__${podSlug}`;
}

function getLegacyCertKey(slug: string): string {
  return `${CERT_KEY_PREFIX}${getCurrentUserId()}_${slug}`;
}

export function getPodCertificate(courseSlug: string, podSlug: string): CertificateData | null {
  if (typeof window === 'undefined') return null;
  try {
    let raw = localStorage.getItem(getPodCertKey(courseSlug, podSlug));
    if (!raw) {
      raw = localStorage.getItem(getLegacyCertKey(podSlug));
    }
    return raw ? (JSON.parse(raw) as CertificateData) : null;
  } catch {
    return null;
  }
}

export function issuePodCertificate(
  courseSlug: string,
  podSlug: string,
  podTitle: string,
  difficulty: 'beginner' | 'intermediate' | 'advanced',
  estimatedHours: number,
  notebookCount: number,
  studentName: string = 'Student'
): CertificateData {
  const existing = getPodCertificate(courseSlug, podSlug);
  if (existing) return existing;

  const cert: CertificateData = {
    certificateId: generateCertificateId(podSlug),
    studentName,
    courseTitle: podTitle,
    courseSlug,
    podSlug,
    completionDate: new Date().toISOString(),
    difficulty,
    estimatedHours,
    notebookCount,
  };
  if (typeof window !== 'undefined') {
    localStorage.setItem(getPodCertKey(courseSlug, podSlug), JSON.stringify(cert));
  }
  return cert;
}

// ─── Course Certificate ──────────────────────────────────────────────

function getCourseCertKey(courseSlug: string): string {
  return `${COURSE_CERT_KEY_PREFIX}${getCurrentUserId()}_${courseSlug}`;
}

export function getCourseCertificate(courseSlug: string): CourseCertificateData | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = localStorage.getItem(getCourseCertKey(courseSlug));
    return raw ? (JSON.parse(raw) as CourseCertificateData) : null;
  } catch {
    return null;
  }
}

export function issueCourseCertificate(
  courseSlug: string,
  courseTitle: string,
  difficulty: 'beginner' | 'intermediate' | 'advanced',
  estimatedHours: number,
  podCount: number,
  totalNotebooks: number,
  studentName: string = 'Student'
): CourseCertificateData {
  const existing = getCourseCertificate(courseSlug);
  if (existing) return existing;

  const cert: CourseCertificateData = {
    certificateId: generateCertificateId(courseSlug),
    studentName,
    courseTitle,
    courseSlug,
    completionDate: new Date().toISOString(),
    difficulty,
    estimatedHours,
    podCount,
    totalNotebooks,
  };
  if (typeof window !== 'undefined') {
    localStorage.setItem(getCourseCertKey(courseSlug), JSON.stringify(cert));
  }
  return cert;
}

// ─── Migration helper ────────────────────────────────────────────────

/**
 * One-time migration of old flat progress keys to new pod-scoped keys.
 * Maps old `vizuara_progress_{uid}_{podSlug}` → new `vizuara_progress_{uid}_{courseSlug}__{podSlug}`.
 */
export function migrateProgressToPodStructure(
  legacyPodMappings: { podSlug: string; courseSlug: string }[]
): void {
  if (typeof window === 'undefined') return;
  const migrationFlag = 'vizuara_pod_migration_v1';
  if (localStorage.getItem(migrationFlag)) return;

  const uid = getCurrentUserId();

  for (const { podSlug, courseSlug } of legacyPodMappings) {
    const oldProgressKey = `${STORAGE_KEY_PREFIX}${uid}_${podSlug}`;
    const newProgressKey = `${STORAGE_KEY_PREFIX}${uid}_${courseSlug}__${podSlug}`;
    const oldProgress = localStorage.getItem(oldProgressKey);
    if (oldProgress && !localStorage.getItem(newProgressKey)) {
      localStorage.setItem(newProgressKey, oldProgress);
    }

    const oldCertKey = `${CERT_KEY_PREFIX}${uid}_${podSlug}`;
    const newCertKey = `${CERT_KEY_PREFIX}${uid}_${courseSlug}__${podSlug}`;
    const oldCert = localStorage.getItem(oldCertKey);
    if (oldCert && !localStorage.getItem(newCertKey)) {
      localStorage.setItem(newCertKey, oldCert);
    }
  }

  localStorage.setItem(migrationFlag, '1');
  notifyListeners();
}

// ─── Utility ─────────────────────────────────────────────────────────

export function clearAllProgress(): void {
  if (typeof window === 'undefined') return;
  const uid = getCurrentUserId();
  const prefixes = [
    `${STORAGE_KEY_PREFIX}${uid}_`,
    `${CERT_KEY_PREFIX}${uid}_`,
    `${COURSE_CERT_KEY_PREFIX}${uid}_`,
  ];
  const keysToRemove: string[] = [];
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (key && prefixes.some((p) => key.startsWith(p))) {
      keysToRemove.push(key);
    }
  }
  keysToRemove.forEach((key) => localStorage.removeItem(key));
  notifyListeners();
}
