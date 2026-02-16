'use client';

interface UserAvatarProps {
  name: string;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

const sizeStyles = {
  sm: 'w-7 h-7 text-xs',
  md: 'w-9 h-9 text-sm',
  lg: 'w-14 h-14 text-lg',
};

export default function UserAvatar({ name, size = 'md', className = '' }: UserAvatarProps) {
  const initials = name
    .split(' ')
    .map((n) => n[0])
    .join('')
    .toUpperCase()
    .slice(0, 2);

  return (
    <div
      className={`
        rounded-full bg-accent-blue text-white font-semibold
        flex items-center justify-center flex-shrink-0
        ${sizeStyles[size]}
        ${className}
      `}
    >
      {initials}
    </div>
  );
}
