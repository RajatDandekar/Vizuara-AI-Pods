export interface User {
  id: string;
  fullName: string;
  email: string;
  experienceLevel: 'beginner' | 'intermediate' | 'advanced' | null;
  onboardingComplete: boolean;
  interests: string[];
}

export interface AuthState {
  user: User | null;
  loading: boolean;
}
