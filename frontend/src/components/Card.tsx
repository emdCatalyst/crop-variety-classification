import { ReactNode } from "react";

export function Card({
  className = "",
  children,
}: {
  className?: string;
  children: ReactNode;
}) {
  return (
    <div
      className={`bg-white rounded-xl border border-slate-200 shadow-sm hover:shadow-md transition-shadow ${className}`}
    >
      {children}
    </div>
  );
}

export function CardHeader({
  title,
  subtitle,
  icon,
  trailing,
}: {
  title: string;
  subtitle?: string;
  icon?: ReactNode;
  trailing?: ReactNode;
}) {
  return (
    <div className="flex items-start justify-between gap-3 p-4 border-b border-slate-100">
      <div className="flex items-start gap-3 min-w-0">
        {icon && (
          <div className="shrink-0 mt-0.5 text-brand-600">{icon}</div>
        )}
        <div className="min-w-0">
          <h3 className="text-sm font-semibold text-slate-900 truncate">{title}</h3>
          {subtitle && (
            <p className="text-xs text-slate-500 mt-0.5">{subtitle}</p>
          )}
        </div>
      </div>
      {trailing && <div className="shrink-0">{trailing}</div>}
    </div>
  );
}

export function CardBody({
  className = "",
  children,
}: {
  className?: string;
  children: ReactNode;
}) {
  return <div className={`p-4 ${className}`}>{children}</div>;
}

export function StatCard({
  label,
  value,
  hint,
  icon,
  accent = "brand",
}: {
  label: string;
  value: number | string;
  hint?: string;
  icon?: ReactNode;
  accent?: "brand" | "amber" | "emerald" | "red" | "slate";
}) {
  const accentBg = {
    brand: "bg-brand-50 text-brand-700",
    amber: "bg-amber-50 text-amber-700",
    emerald: "bg-emerald-50 text-emerald-700",
    red: "bg-red-50 text-red-700",
    slate: "bg-slate-100 text-slate-700",
  }[accent];

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm hover:shadow-md transition-shadow p-4 flex items-center gap-3">
      {icon && (
        <div
          className={`shrink-0 w-10 h-10 rounded-lg ${accentBg} flex items-center justify-center`}
        >
          {icon}
        </div>
      )}
      <div className="min-w-0">
        <div className="text-xs font-medium text-slate-500 uppercase tracking-wide truncate">
          {label}
        </div>
        <div className="text-2xl font-bold text-slate-900 mt-0.5 leading-tight">
          {value}
        </div>
        {hint && <div className="text-xs text-slate-400 mt-0.5">{hint}</div>}
      </div>
    </div>
  );
}
