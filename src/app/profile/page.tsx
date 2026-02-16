'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/context/AuthContext';
import Button from '@/components/ui/Button';
import Badge from '@/components/ui/Badge';
import UserAvatar from '@/components/auth/UserAvatar';
import FadeIn from '@/components/animations/FadeIn';
import catalogData from '@/../content/courses/catalog.json';

const ALL_INTEREST_TAGS = Array.from(
  new Set(
    catalogData.courses.flatMap((c: { tags: string[] }) => c.tags).filter(Boolean)
  )
).sort() as string[];

const TAG_LABELS: Record<string, string> = {
  'robotics': 'Robotics',
  'nlp': 'NLP',
  'transformers': 'Transformers',
  'reinforcement-learning': 'Reinforcement Learning',
  'language-models': 'Language Models',
  'computer-vision': 'Computer Vision',
  'distributed-training': 'Distributed Training',
  'foundation-models': 'Foundation Models',
  'diffusion-models': 'Diffusion Models',
  'agents': 'AI Agents',
  'reasoning': 'Reasoning',
  'efficiency': 'Efficiency',
  'graph-networks': 'Graph Networks',
  'alignment': 'AI Alignment',
  'rag': 'RAG',
  'world-models': 'World Models',
  'simulation': 'Simulation',
  'vision-language': 'Vision-Language',
  'systems': 'Systems',
  'gpu': 'GPU Programming',
  'geometric-dl': 'Geometric DL',
};

export default function ProfilePage() {
  const { user, logout, refreshUser } = useAuth();
  const router = useRouter();

  const [editingName, setEditingName] = useState(false);
  const [fullName, setFullName] = useState('');
  const [editingInterests, setEditingInterests] = useState(false);
  const [selectedInterests, setSelectedInterests] = useState<string[]>([]);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (user) {
      setFullName(user.fullName);
      setSelectedInterests(user.interests);
    }
  }, [user]);

  useEffect(() => {
    if (user === null) {
      router.push('/auth/login');
    }
  }, [user, router]);

  if (!user) return null;

  function toggleInterest(tag: string) {
    setSelectedInterests((prev) =>
      prev.includes(tag) ? prev.filter((t) => t !== tag) : [...prev, tag]
    );
  }

  async function saveName() {
    setSaving(true);
    await fetch('/api/auth/profile', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fullName }),
    });
    await refreshUser();
    setEditingName(false);
    setSaving(false);
  }

  async function saveInterests() {
    setSaving(true);
    await fetch('/api/auth/profile', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ interests: selectedInterests }),
    });
    await refreshUser();
    setEditingInterests(false);
    setSaving(false);
  }

  async function handleLogout() {
    await logout();
    router.push('/');
  }

  return (
    <div className="max-w-2xl mx-auto px-4 sm:px-6 py-12">
      <FadeIn>
        <h1 className="text-2xl font-bold text-foreground mb-8">My Profile</h1>

        {/* Avatar + Name section */}
        <div className="bg-card-bg border border-card-border rounded-2xl p-6 mb-6">
          <div className="flex items-center gap-4 mb-6">
            <UserAvatar name={user.fullName} size="lg" />
            <div>
              {editingName ? (
                <div className="flex items-center gap-2">
                  <input
                    type="text"
                    value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                    className="px-3 py-1.5 text-sm border border-card-border rounded-lg focus:outline-none focus:ring-2 focus:ring-accent-blue/30 focus:border-accent-blue"
                  />
                  <Button size="sm" onClick={saveName} isLoading={saving}>Save</Button>
                  <Button size="sm" variant="ghost" onClick={() => { setEditingName(false); setFullName(user.fullName); }}>
                    Cancel
                  </Button>
                </div>
              ) : (
                <div className="flex items-center gap-2">
                  <h2 className="text-lg font-semibold text-foreground">{user.fullName}</h2>
                  <button
                    onClick={() => setEditingName(true)}
                    className="text-text-muted hover:text-accent-blue transition-colors cursor-pointer"
                  >
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M16.862 4.487l1.687-1.688a1.875 1.875 0 112.652 2.652L10.582 16.07a4.5 4.5 0 01-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 011.13-1.897l8.932-8.931z" />
                    </svg>
                  </button>
                </div>
              )}
              <p className="text-sm text-text-muted">{user.email}</p>
              {user.experienceLevel && (
                <div className="mt-1">
                  <Badge variant="blue" size="sm">
                    {user.experienceLevel}
                  </Badge>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Interests section */}
        <div id="interests" className="bg-card-bg border border-card-border rounded-2xl p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-foreground">My Interests</h3>
            {!editingInterests && (
              <button
                onClick={() => setEditingInterests(true)}
                className="text-sm text-accent-blue hover:underline cursor-pointer"
              >
                Edit
              </button>
            )}
          </div>

          {editingInterests ? (
            <>
              <div className="flex flex-wrap gap-2 mb-4">
                {ALL_INTEREST_TAGS.map((tag) => (
                  <button
                    key={tag}
                    onClick={() => toggleInterest(tag)}
                    className={`
                      px-3.5 py-2 rounded-xl text-sm font-medium transition-all duration-200 cursor-pointer
                      ${selectedInterests.includes(tag)
                        ? 'bg-accent-blue text-white shadow-sm'
                        : 'bg-gray-100 text-text-secondary hover:bg-gray-200'
                      }
                    `}
                  >
                    {TAG_LABELS[tag] || tag}
                  </button>
                ))}
              </div>
              <div className="flex gap-2">
                <Button size="sm" onClick={saveInterests} isLoading={saving}>Save Interests</Button>
                <Button size="sm" variant="ghost" onClick={() => { setEditingInterests(false); setSelectedInterests(user.interests); }}>
                  Cancel
                </Button>
              </div>
            </>
          ) : (
            <div className="flex flex-wrap gap-2">
              {user.interests.length > 0 ? (
                user.interests.map((tag) => (
                  <Badge key={tag} variant="blue" size="md">
                    {TAG_LABELS[tag] || tag}
                  </Badge>
                ))
              ) : (
                <p className="text-sm text-text-muted">
                  No interests selected yet.{' '}
                  <button onClick={() => setEditingInterests(true)} className="text-accent-blue hover:underline cursor-pointer">
                    Add some
                  </button>
                </p>
              )}
            </div>
          )}
        </div>

        {/* Logout */}
        <div className="bg-card-bg border border-card-border rounded-2xl p-6">
          <Button variant="secondary" onClick={handleLogout}>
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 9V5.25A2.25 2.25 0 0013.5 3h-6a2.25 2.25 0 00-2.25 2.25v13.5A2.25 2.25 0 007.5 21h6a2.25 2.25 0 002.25-2.25V15m3 0l3-3m0 0l-3-3m3 3H9" />
            </svg>
            Log Out
          </Button>
        </div>
      </FadeIn>
    </div>
  );
}
