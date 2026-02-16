import { CourseProvider } from '@/context/CourseContext';
import ErrorBoundary from '@/components/ui/ErrorBoundary';

export default function DiscoverLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <ErrorBoundary>
      <CourseProvider>{children}</CourseProvider>
    </ErrorBoundary>
  );
}
