import type { ReactNode } from 'react';

interface InfoCardProps {
  tone: 'brand' | 'teal' | 'purple';
  icon: ReactNode;
  title: string;
  children: ReactNode;
}

export function InfoCard({ tone, icon, title, children }: InfoCardProps) {
  return (
    <article className="info-card surface" data-tone={tone}>
      <span className="info-icon" aria-hidden="true">
        {icon}
      </span>
      <h2 className="info-title">{title}</h2>
      <p className="info-text">{children}</p>
    </article>
  );
}
