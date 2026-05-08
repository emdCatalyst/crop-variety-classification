import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  AdminActivityRow,
  AdminStats,
  getActivity,
  getStats,
} from "@/api/admin";

export default function AdminDashboardPage() {
  const { t, i18n } = useTranslation();
  const [stats, setStats] = useState<AdminStats | null>(null);
  const [activity, setActivity] = useState<AdminActivityRow[]>([]);
  const [loading, setLoading] = useState(true);
  const locale = i18n.language.split("-")[0];

  useEffect(() => {
    Promise.all([getStats(), getActivity()])
      .then(([s, a]) => {
        setStats(s);
        setActivity(a);
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading || !stats) {
    return <p className="text-sm text-slate-500">{t("common.loading")}</p>;
  }

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold">{t("admin.title")}</h1>
        <p className="text-sm text-slate-600 mt-1">{t("admin.subtitle")}</p>
      </header>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Stat label={t("admin.stats.users")} value={stats.users_total} hint={`${stats.users_active} ${t("admin.stats.active")}`} />
        <Stat label={t("admin.stats.admins")} value={stats.admins} />
        <Stat label={t("admin.stats.analyses")} value={stats.analyses_total} />
        <Stat label={t("admin.stats.notifications")} value={stats.notifications_total} />
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Stat label={t("status.queued")} value={stats.analyses_by_status.queued} />
        <Stat label={t("status.processing")} value={stats.analyses_by_status.processing} />
        <Stat label={t("status.completed")} value={stats.analyses_by_status.completed} />
        <Stat label={t("status.failed")} value={stats.analyses_by_status.failed} />
      </div>

      <section className="bg-white rounded-lg shadow-sm">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500 p-3 border-b border-slate-200">
          {t("admin.recent_activity")}
        </h2>
        {activity.length === 0 ? (
          <p className="p-4 text-sm text-slate-500">{t("admin.no_activity")}</p>
        ) : (
          <ul className="divide-y divide-slate-200">
            {activity.map((a, i) => (
              <li key={i} className="p-3 flex items-start justify-between gap-3 text-sm">
                <div className="min-w-0">
                  <div className="font-medium truncate">{a.title}</div>
                  {a.detail && (
                    <div className="text-xs text-slate-500 truncate">{a.detail}</div>
                  )}
                </div>
                <div className="text-xs text-slate-400 shrink-0">
                  {new Date(a.at).toLocaleString(locale)}
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

function Stat({ label, value, hint }: { label: string; value: number; hint?: string }) {
  return (
    <div className="bg-white rounded-lg shadow-sm p-4">
      <div className="text-xs text-slate-500">{label}</div>
      <div className="text-2xl font-bold mt-1">{value}</div>
      {hint && <div className="text-[10px] text-slate-400 mt-1">{hint}</div>}
    </div>
  );
}
