'use client';

import { useState, useEffect, useCallback } from 'react';
import type { FeedbackRecord, FeedbackStats } from '@/types/feedback';

export default function AdminFeedbackDashboard() {
  const [feedback, setFeedback] = useState<FeedbackRecord[]>([]);
  const [stats, setStats] = useState<FeedbackStats | null>(null);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);

  // Filters
  const [typeFilter, setTypeFilter] = useState('');
  const [courseFilter, setCourseFilter] = useState('');
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');

  const fetchData = useCallback(async () => {
    setLoading(true);
    const params = new URLSearchParams();
    params.set('page', String(page));
    if (typeFilter) params.set('type', typeFilter);
    if (courseFilter) params.set('courseSlug', courseFilter);
    if (dateFrom) params.set('dateFrom', dateFrom);
    if (dateTo) params.set('dateTo', dateTo);

    const res = await fetch(`/api/admin/feedback?${params}`);
    if (res.ok) {
      const data = await res.json();
      setFeedback(data.feedback);
      setStats(data.stats);
      setTotalPages(data.totalPages);
      setTotal(data.total);
    }
    setLoading(false);
  }, [page, typeFilter, courseFilter, dateFrom, dateTo]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  function handleExport() {
    const params = new URLSearchParams();
    if (typeFilter) params.set('type', typeFilter);
    if (courseFilter) params.set('courseSlug', courseFilter);
    if (dateFrom) params.set('dateFrom', dateFrom);
    if (dateTo) params.set('dateTo', dateTo);
    window.open(`/api/admin/feedback/export?${params}`, '_blank');
  }

  const maxNps = stats ? Math.max(...Object.values(stats.npsDistribution), 1) : 1;

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 py-8">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-foreground">Feedback Dashboard</h1>
        <button
          onClick={handleExport}
          className="text-sm px-4 py-2 rounded-lg bg-foreground text-white hover:bg-gray-800 transition-colors cursor-pointer"
        >
          Export CSV
        </button>
      </div>

      {/* Stats bar */}
      {stats && (
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 mb-6">
          <StatCard label="Total Feedback" value={String(stats.totalCount)} />
          <StatCard label="Avg NPS" value={stats.avgNps != null ? stats.avgNps.toFixed(1) : '—'} />
          <StatCard label="Avg Emoji" value={stats.avgEmoji != null ? stats.avgEmoji.toFixed(1) + '/5' : '—'} />
          <StatCard label="Thumbs Up %" value={stats.thumbsUpPercent != null ? stats.thumbsUpPercent.toFixed(0) + '%' : '—'} />
          <StatCard label="Last 7 Days" value={String(stats.last7DaysCount)} />
        </div>
      )}

      {/* NPS Distribution + Tag Breakdown */}
      {stats && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6">
          <div className="border border-card-border rounded-xl p-4 bg-card-bg">
            <h3 className="text-sm font-semibold text-foreground mb-3">NPS Distribution</h3>
            <div className="flex items-end gap-1 h-24">
              {Array.from({ length: 10 }, (_, i) => i + 1).map(score => {
                const count = stats.npsDistribution[score] || 0;
                const height = maxNps > 0 ? (count / maxNps) * 100 : 0;
                return (
                  <div key={score} className="flex-1 flex flex-col items-center gap-1">
                    <div
                      className={`w-full rounded-t transition-all ${
                        score >= 9 ? 'bg-accent-green' : score >= 7 ? 'bg-accent-amber' : 'bg-red-400'
                      }`}
                      style={{ height: `${Math.max(height, 2)}%` }}
                    />
                    <span className="text-[10px] text-text-muted">{score}</span>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="border border-card-border rounded-xl p-4 bg-card-bg">
            <h3 className="text-sm font-semibold text-foreground mb-3">Tag Breakdown</h3>
            <div className="space-y-2">
              {Object.entries(stats.tagBreakdown)
                .sort((a, b) => b[1] - a[1])
                .map(([tag, count]) => (
                  <div key={tag} className="flex items-center gap-2 text-sm">
                    <span className="text-text-secondary flex-1">{tag.replace(/_/g, ' ')}</span>
                    <span className="font-medium text-foreground">{count}</span>
                  </div>
                ))}
              {Object.keys(stats.tagBreakdown).length === 0 && (
                <p className="text-xs text-text-muted">No tags yet</p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-3 mb-4">
        <select
          value={typeFilter}
          onChange={e => { setTypeFilter(e.target.value); setPage(1); }}
          className="text-sm border border-gray-200 rounded-lg px-3 py-1.5 bg-white"
        >
          <option value="">All Types</option>
          <option value="emoji">Emoji</option>
          <option value="nps">NPS</option>
          <option value="thumbs">Thumbs</option>
          <option value="survey">Survey</option>
          <option value="general">General</option>
        </select>

        <input
          type="text"
          placeholder="Course slug"
          value={courseFilter}
          onChange={e => { setCourseFilter(e.target.value); setPage(1); }}
          className="text-sm border border-gray-200 rounded-lg px-3 py-1.5 bg-white w-40"
        />

        <input
          type="date"
          value={dateFrom}
          onChange={e => { setDateFrom(e.target.value); setPage(1); }}
          className="text-sm border border-gray-200 rounded-lg px-3 py-1.5 bg-white"
        />
        <span className="text-xs text-text-muted">to</span>
        <input
          type="date"
          value={dateTo}
          onChange={e => { setDateTo(e.target.value); setPage(1); }}
          className="text-sm border border-gray-200 rounded-lg px-3 py-1.5 bg-white"
        />

        <span className="text-xs text-text-muted ml-auto">{total} results</span>
      </div>

      {/* Table */}
      <div className="border border-card-border rounded-xl overflow-hidden bg-card-bg">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-card-border bg-gray-50/50">
                <th className="text-left py-2.5 px-3 font-medium text-text-secondary">Date</th>
                <th className="text-left py-2.5 px-3 font-medium text-text-secondary">User</th>
                <th className="text-left py-2.5 px-3 font-medium text-text-secondary">Type</th>
                <th className="text-left py-2.5 px-3 font-medium text-text-secondary">Course / Pod</th>
                <th className="text-left py-2.5 px-3 font-medium text-text-secondary">Rating</th>
                <th className="text-left py-2.5 px-3 font-medium text-text-secondary">Details</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr><td colSpan={6} className="text-center py-8 text-text-muted">Loading...</td></tr>
              ) : feedback.length === 0 ? (
                <tr><td colSpan={6} className="text-center py-8 text-text-muted">No feedback found</td></tr>
              ) : feedback.map(f => (
                <tr key={f.id} className="border-b border-card-border last:border-0 hover:bg-gray-50/50">
                  <td className="py-2.5 px-3 text-xs text-text-muted whitespace-nowrap">
                    {new Date(f.createdAt).toLocaleDateString()}
                  </td>
                  <td className="py-2.5 px-3">
                    <div className="text-xs font-medium text-foreground">{f.userName}</div>
                    <div className="text-[10px] text-text-muted">{f.userEmail}</div>
                  </td>
                  <td className="py-2.5 px-3">
                    <TypeBadge type={f.type} />
                  </td>
                  <td className="py-2.5 px-3 text-xs text-text-secondary">
                    {f.courseSlug && <span>{f.courseSlug}</span>}
                    {f.podSlug && <span className="text-text-muted"> / {f.podSlug}</span>}
                    {f.contentType && <span className="text-text-muted"> ({f.contentType})</span>}
                  </td>
                  <td className="py-2.5 px-3">
                    <RatingDisplay type={f.type} rating={f.rating} />
                  </td>
                  <td className="py-2.5 px-3 text-xs text-text-secondary min-w-[200px] max-w-md">
                    {f.comment && <p className="whitespace-pre-wrap break-words">{f.comment}</p>}
                    {f.tags && f.tags.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-1">
                        {f.tags.map(tag => (
                          <span key={tag} className="text-[10px] px-1.5 py-0.5 bg-gray-100 rounded text-text-muted">
                            {tag.replace(/_/g, ' ')}
                          </span>
                        ))}
                      </div>
                    )}
                    {f.surveyData && (
                      <div className="text-[10px] text-text-muted mt-1">
                        {Object.entries(f.surveyData).map(([k, v]) => (
                          <span key={k} className="mr-2">{k}: {v}</span>
                        ))}
                      </div>
                    )}
                    {f.category && <span className="text-[10px] text-text-muted">[{f.category}]</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-2 mt-4">
          <button
            onClick={() => setPage(p => Math.max(1, p - 1))}
            disabled={page === 1}
            className="text-sm px-3 py-1.5 rounded-lg border border-gray-200 hover:bg-gray-50 disabled:opacity-50 cursor-pointer"
          >
            Previous
          </button>
          <span className="text-sm text-text-muted">
            Page {page} of {totalPages}
          </span>
          <button
            onClick={() => setPage(p => Math.min(totalPages, p + 1))}
            disabled={page === totalPages}
            className="text-sm px-3 py-1.5 rounded-lg border border-gray-200 hover:bg-gray-50 disabled:opacity-50 cursor-pointer"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="border border-card-border rounded-xl p-3 bg-card-bg">
      <p className="text-xs text-text-muted mb-0.5">{label}</p>
      <p className="text-lg font-semibold text-foreground">{value}</p>
    </div>
  );
}

function TypeBadge({ type }: { type: string }) {
  const colors: Record<string, string> = {
    emoji: 'bg-amber-50 text-amber-600',
    nps: 'bg-blue-50 text-blue-600',
    thumbs: 'bg-green-50 text-green-600',
    survey: 'bg-purple-50 text-purple-600',
    general: 'bg-gray-100 text-gray-600',
  };
  return (
    <span className={`text-[10px] font-medium px-2 py-0.5 rounded-full ${colors[type] || colors.general}`}>
      {type}
    </span>
  );
}

function RatingDisplay({ type, rating }: { type: string; rating: number | null }) {
  if (rating === null) return <span className="text-text-muted text-xs">—</span>;

  if (type === 'emoji') {
    const emojis = ['', '😕', '😐', '🙂', '😊', '🤩'];
    return <span className="text-base">{emojis[rating] || rating}</span>;
  }
  if (type === 'thumbs') {
    return <span className="text-base">{rating === 1 ? '👍' : '👎'}</span>;
  }
  if (type === 'nps') {
    return (
      <span className={`text-xs font-medium ${
        rating >= 9 ? 'text-accent-green' : rating >= 7 ? 'text-accent-amber' : 'text-red-500'
      }`}>
        {rating}/10
      </span>
    );
  }
  return <span className="text-xs">{rating}</span>;
}
