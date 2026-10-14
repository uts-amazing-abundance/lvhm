import { cn } from 'cn';
import type { ReactNode } from 'react';
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

interface Props {
  title: string;
  description?: ReactNode;
  action?: ReactNode;
  className?: string;
  contentClassName?: string;
  children: ReactNode;
}

/** A dashboard section: shadcn Card with a level-2 heading, optional note and header action. */
export function SectionCard({ title, description, action, className, contentClassName, children }: Props) {
  return (
    <Card className={cn('mb-4 gap-3 ring-line', className)}>
      <CardHeader>
        <CardTitle role="heading" aria-level={2} className="font-semibold">{title}</CardTitle>
        {description && <CardDescription className="text-[13px] text-ink-3">{description}</CardDescription>}
        {action && <CardAction>{action}</CardAction>}
      </CardHeader>
      <CardContent className={contentClassName}>{children}</CardContent>
    </Card>
  );
}
