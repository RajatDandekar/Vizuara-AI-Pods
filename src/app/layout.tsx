import type { Metadata } from 'next';
import { Figtree, JetBrains_Mono } from 'next/font/google';
import './globals.css';
import Header from '@/components/layout/Header';
import Footer from '@/components/layout/Footer';
import { AuthProvider } from '@/context/AuthContext';
import { SubscriptionProvider } from '@/context/SubscriptionContext';
import { ThemeProvider } from '@/context/ThemeContext';
import FeedbackTab from '@/components/feedback/FeedbackTab';

const figtree = Figtree({
  variable: '--font-figtree',
  subsets: ['latin'],
  display: 'swap',
});

const jetbrainsMono = JetBrains_Mono({
  variable: '--font-jetbrains',
  subsets: ['latin'],
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'Vizuara AI Pods — Learn AI from First Principles',
  description:
    'The FPR Technique: Fundamentals, Practicals and Research. Master AI through curated articles, hands-on Colab notebooks, and real-world case studies.',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `(function(){try{var t=localStorage.getItem('vizuara-theme');if(t==='dark')document.documentElement.classList.add('dark')}catch(e){}})()`,
          }}
        />
      </head>
      <body
        className={`${figtree.variable} ${jetbrainsMono.variable} antialiased min-h-screen flex flex-col font-light`}
        style={{ fontFamily: 'var(--font-figtree), system-ui, sans-serif' }}
      >
        <AuthProvider>
          <SubscriptionProvider>
            <ThemeProvider>
              <Header />
              <main className="flex-1">{children}</main>
              <Footer />
              <FeedbackTab />
            </ThemeProvider>
          </SubscriptionProvider>
        </AuthProvider>
      </body>
    </html>
  );
}
