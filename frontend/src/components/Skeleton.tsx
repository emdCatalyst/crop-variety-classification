/**
 * Pulse-animated placeholder for loading states. Use width/height via Tailwind:
 *   <Skeleton className="h-4 w-24" />
 *   <SkeletonRow />  ← convenience for list rows
 */
export function Skeleton({
  className = "",
  style,
}: {
  className?: string;
  style?: React.CSSProperties;
}) {
  return (
    <div
      className={`bg-slate-200/70 rounded-md animate-pulse ${className}`}
      style={style}
      aria-hidden
    />
  );
}

export function SkeletonRow({ lines = 2 }: { lines?: number }) {
  return (
    <div className="p-4 flex items-center justify-between gap-3">
      <div className="min-w-0 flex-1 space-y-2">
        <Skeleton className="h-4 w-2/3" />
        {Array.from({ length: lines - 1 }).map((_, i) => (
          <Skeleton key={i} className="h-3 w-1/3" />
        ))}
      </div>
      <Skeleton className="h-6 w-16 shrink-0" />
    </div>
  );
}

export function SkeletonStat() {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-4 flex items-center gap-3">
      <Skeleton className="w-10 h-10 rounded-lg" />
      <div className="flex-1 space-y-2">
        <Skeleton className="h-3 w-16" />
        <Skeleton className="h-6 w-12" />
      </div>
    </div>
  );
}

export function SkeletonChart({ height = 220 }: { height?: number }) {
  // Pseudo-random but stable bar heights so it actually looks like a chart.
  const heights = [60, 80, 50, 90, 70, 95, 55, 85, 72, 65, 88, 60];
  return (
    <div
      className="grid grid-cols-12 gap-1.5 items-end"
      style={{ height }}
      aria-hidden
    >
      {heights.map((h, i) => (
        <Skeleton key={i} className="w-full" style={{ height: `${h}%` }} />
      ))}
    </div>
  );
}
