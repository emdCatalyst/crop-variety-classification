import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  Activity,
  Bell,
  CheckCircle2,
  Clock,
  ListChecks,
  ShieldCheck,
  Users,
  XCircle,
} from "lucide-react";
import {
  AdminActivityRow,
  AdminStats,
  AdminTimeseriesPoint,
  getActivity,
  getAdminTimeseries,
  getStats,
} from "@/api/admin";
import { Card, CardBody, CardHeader, StatCard } from "@/components/Card";
import AdminTimeseries from "@/components/charts/AdminTimeseries";
import StatusBreakdown from "@/components/charts/StatusBreakdown";
import { SkeletonChart, SkeletonRow, SkeletonStat } from "@/components/Skeleton";

export default function AdminDashboardPage() {
  const { t, i18n } = useTranslation();
  const [stats, setStats] = useState<AdminStats | null>(null);
  const [activity, setActivity] = useState<AdminActivityRow[]>([]);
  const [series, setSeries] = useState<AdminTimeseriesPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const locale = i18n.language.split("-")[0];

  useEffect(() => {
    Promise.all([getStats(), getActivity(), getAdminTimeseries(30)])
      .then(([s, a, ts]) => {
        setStats(s);
        setActivity(a);
        setSeries(ts);
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading || !stats) {
    return (
      <div className="space-y-6 animate-fade-in">
        <header>
          <h1 className="text-2xl font-bold text-slate-900">{t("admin.title")}</h1>
          <p className="text-sm text-slate-500 mt-1">{t("admin.subtitle")}</p>
        </header>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {Array.from({ length: 4 }).map((_, i) => (
            <SkeletonStat key={i} />
          ))}
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {Array.from({ length: 4 }).map((_, i) => (
            <SkeletonStat key={i} />
          ))}
        </div>
        <div className="grid gap-4 md:grid-cols-3">
          <Card className="md:col-span-2">
            <CardBody>
              <SkeletonChart height={240} />
            </CardBody>
          </Card>
          <Card>
            <CardBody>
              <SkeletonChart height={160} />
            </CardBody>
          </Card>
        </div>
        <Card>
          <ul className="divide-y divide-slate-100">
            {Array.from({ length: 4 }).map((_, i) => (
              <li key={i}>
                <SkeletonRow />
              </li>
            ))}
          </ul>
        </Card>
      </div>
    );
  }

  const slices = [
    { key: "queued", label: t("status.queued"), value: stats.analyses_by_status.queued },
    {
      key: "processing",
      label: t("status.processing"),
      value: stats.analyses_by_status.processing,
    },
    {
      key: "completed",
      label: t("status.completed"),
      value: stats.analyses_by_status.completed,
    },
    { key: "failed", label: t("status.failed"), value: stats.analyses_by_status.failed },
  ];

  return (
    <div className="space-y-6 animate-fade-in">
      <header>
        <h1 className="text-2xl font-bold text-slate-900">{t("admin.title")}</h1>
        <p className="text-sm text-slate-500 mt-1">{t("admin.subtitle")}</p>
      </header>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard
          label={t("admin.stats.users")}
          value={stats.users_total}
          hint={`${stats.users_active} ${t("admin.stats.active")}`}
          icon={<Users size={20} />}
          accent="brand"
        />
        <StatCard
          label={t("admin.stats.admins")}
          value={stats.admins}
          icon={<ShieldCheck size={20} />}
          accent="slate"
        />
        <StatCard
          label={t("admin.stats.analyses")}
          value={stats.analyses_total}
          icon={<ListChecks size={20} />}
          accent="emerald"
        />
        <StatCard
          label={t("admin.stats.notifications")}
          value={stats.notifications_total}
          icon={<Bell size={20} />}
          accent="amber"
        />
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard
          label={t("status.queued")}
          value={stats.analyses_by_status.queued}
          icon={<Clock size={20} />}
          accent="slate"
        />
        <StatCard
          label={t("status.processing")}
          value={stats.analyses_by_status.processing}
          icon={<Clock size={20} />}
          accent="amber"
        />
        <StatCard
          label={t("status.completed")}
          value={stats.analyses_by_status.completed}
          icon={<CheckCircle2 size={20} />}
          accent="emerald"
        />
        <StatCard
          label={t("status.failed")}
          value={stats.analyses_by_status.failed}
          icon={<XCircle size={20} />}
          accent="red"
        />
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <Card className="md:col-span-2">
          <CardHeader
            title={t("admin.charts.timeseries")}
            subtitle={t("admin.charts.timeseries_hint")}
            icon={<Activity size={18} />}
          />
          <CardBody>
            <AdminTimeseries
              data={series}
              locale={locale}
              labels={{
                analyses: t("admin.charts.analyses"),
                new_users: t("admin.charts.new_users"),
              }}
            />
          </CardBody>
        </Card>

        <Card>
          <CardHeader title={t("admin.charts.status_breakdown")} />
          <CardBody>
            <StatusBreakdown slices={slices} />
          </CardBody>
        </Card>
      </div>

      <Card>
        <CardHeader title={t("admin.recent_activity")} />
        {activity.length === 0 ? (
          <CardBody>
            <p className="text-sm text-slate-500">{t("admin.no_activity")}</p>
          </CardBody>
        ) : (
          <ul className="divide-y divide-slate-100">
            {activity.map((a, i) => (
              <li
                key={i}
                className="px-4 py-3 flex items-start justify-between gap-3 text-sm"
              >
                <div className="min-w-0">
                  <div className="font-medium text-slate-900 truncate">{a.title}</div>
                  {a.detail && (
                    <div className="text-xs text-slate-500 truncate">{a.detail}</div>
                  )}
                </div>
                <div className="text-xs text-slate-400 shrink-0 tabular-nums">
                  {new Date(a.at).toLocaleString(locale)}
                </div>
              </li>
            ))}
          </ul>
        )}
      </Card>
    </div>
  );
}
