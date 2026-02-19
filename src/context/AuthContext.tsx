'use client';

import { createContext, useContext, useReducer, useEffect, useCallback, type ReactNode } from 'react';
import type { User, AuthState } from '@/types/auth';
import { setProgressUser } from '@/lib/progress';

const VIZUARA_URL = process.env.NEXT_PUBLIC_VIZUARA_URL || 'https://vizuara.ai';

type AuthAction =
  | { type: 'SET_USER'; user: User | null }
  | { type: 'SET_LOADING'; loading: boolean }
  | { type: 'UPDATE_USER'; updates: Partial<User> };

function authReducer(state: AuthState, action: AuthAction): AuthState {
  switch (action.type) {
    case 'SET_USER':
      return { user: action.user, loading: false };
    case 'SET_LOADING':
      return { ...state, loading: action.loading };
    case 'UPDATE_USER':
      if (!state.user) return state;
      return { ...state, user: { ...state.user, ...action.updates } };
    default:
      return state;
  }
}

interface AuthContextValue {
  user: User | null;
  loading: boolean;
  logout: () => Promise<void>;
  refreshUser: () => Promise<void>;
  updateUser: (updates: Partial<User>) => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(authReducer, { user: null, loading: true });

  const refreshUser = useCallback(async () => {
    try {
      const res = await fetch('/api/auth/me');
      const data = await res.json();
      setProgressUser(data.user?.id ?? null);
      dispatch({ type: 'SET_USER', user: data.user });
    } catch {
      setProgressUser(null);
      dispatch({ type: 'SET_USER', user: null });
    }
  }, []);

  useEffect(() => {
    refreshUser();
  }, [refreshUser]);

  useEffect(() => {
    setProgressUser(state.user?.id ?? null);
  }, [state.user?.id]);

  const logout = useCallback(async () => {
    await fetch('/api/auth/logout', { method: 'POST' });
    setProgressUser(null);
    dispatch({ type: 'SET_USER', user: null });
    window.location.href = VIZUARA_URL;
  }, []);

  const updateUser = useCallback((updates: Partial<User>) => {
    dispatch({ type: 'UPDATE_USER', updates });
  }, []);

  return (
    <AuthContext.Provider
      value={{
        user: state.user,
        loading: state.loading,
        logout,
        refreshUser,
        updateUser,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
