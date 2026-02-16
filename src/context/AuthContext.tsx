'use client';

import { createContext, useContext, useReducer, useEffect, useCallback, type ReactNode } from 'react';
import type { User, AuthState } from '@/types/auth';
import { setProgressUser } from '@/lib/progress';

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
  login: (email: string, password: string, rememberMe?: boolean) => Promise<{ error?: string }>;
  signup: (fullName: string, email: string, password: string, confirmPassword: string) => Promise<{ error?: string }>;
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

  // Keep progress system in sync whenever user state changes
  // (also handles dev server hot reloads where module state resets)
  useEffect(() => {
    setProgressUser(state.user?.id ?? null);
  }, [state.user?.id]);

  const login = useCallback(async (email: string, password: string, rememberMe = false) => {
    const res = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password, rememberMe }),
    });
    const data = await res.json();
    if (!res.ok) return { error: data.error };
    setProgressUser(data.user.id);
    dispatch({ type: 'SET_USER', user: data.user });
    return {};
  }, []);

  const signup = useCallback(async (fullName: string, email: string, password: string, confirmPassword: string) => {
    const res = await fetch('/api/auth/signup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fullName, email, password, confirmPassword }),
    });
    const data = await res.json();
    if (!res.ok) return { error: data.error };
    setProgressUser(data.user.id);
    dispatch({ type: 'SET_USER', user: data.user });
    return {};
  }, []);

  const logout = useCallback(async () => {
    await fetch('/api/auth/logout', { method: 'POST' });
    setProgressUser(null);
    dispatch({ type: 'SET_USER', user: null });
  }, []);

  const updateUser = useCallback((updates: Partial<User>) => {
    dispatch({ type: 'UPDATE_USER', updates });
  }, []);

  return (
    <AuthContext.Provider
      value={{
        user: state.user,
        loading: state.loading,
        login,
        signup,
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
