import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import {
  Activity,
  CheckCircle2,
  Clock,
  FileBarChart,
  Plus,
  Sprout,
  XCircle,
} from "lucide-react";
import {
  Analysis,
  AnalysisDailyPoint,
  api,
  fetchAnalysesTimeseries,
} from "@/api/client";
import { useNotificationsStream } from "@/hooks/useNotificationsStream";
import { Card, CardBody, CardHeader, StatCard } from "@/components/Card";
import AnalysesOverTime from "@/components/charts/AnalysesOverTime";
import StatusBreakdown from "@/components/charts/StatusBreakdown";
import { SkeletonChart, SkeletonRow } from "@/components/Skeleton";

const STATUS_DOTS: Record<string, string> = {
  queued: "bg-slate-400",
  processing: "bg-amber-500",
  completed: "bg-emerald-500",
  failed: "bg-red-500",
};

export default function DashboardPage() {
  const { t, i18n } = useTranslation();
  const [items, setItems] = useState<Analysis[]>([]);
  const [series, setSeries] = useState<AnalysisDailyPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const locale = i18n.language.split("-")[0];

  const refresh = useCallback(() => {
    Promise.all([api.get<Analysis[]>("/analyses"), fetchAnalysesTimeseries(30)])
      .then(([r, ts]) => {
        setItems(r.data);
        setSeries(ts);
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useNotificationsStream(refresh);

  const total = items.length;
  const queued = items.filter((a) => a.status === "queued").length;
  const processing = items.filter((a) => a.status === "processing").length;
  const completed = items.filter((a) => a.status === "completed").length;
  const failed = items.filter((a) => a.status === "failed").length;

  const slices = [
    { key: "queued", label: t("status.queued"), value: queued },
    { key: "processing", label: t("status.processing"), value: processing },
    { key: "completed", label: t("status.completed"), value: completed },
    { key: "failed", label: t("status.failed"), value: failed },
  ];

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-end justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">
            {t("dashboard.title")}
          </h1>
          <p className="text-sm text-slate-500 mt-1">{t("dashboard.subtitle")}</p>
        </div>
        <Link
          to="/upload"
          className="inline-flex items-center gap-1.5 rounded-md bg-brand-600 hover:bg-brand-700 text-white px-4 py-2 text-sm font-semibold shadow-sm"
        >
          <Plus size={16} aria-hidden />
          {t("dashboard.new_analysis")}
        </Link>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard
          label={t("dashboard.total_analyses")}
          value={total}
          icon={<FileBarChart size={20} />}
          accent="brand"
        />
        <StatCard
          label={t("status.processing")}
          value={queued + processing}
          icon={<Clock size={20} />}
          accent="amber"
        />
        <StatCard
          label={t("status.completed")}
          value={completed}
          icon={<CheckCircle2 size={20} />}
          accent="emerald"
        />
        <StatCard
          label={t("status.failed")}
          value={failed}
          icon={<XCircle size={20} />}
          accent="red"
        />
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <Card className="md:col-span-2">
          <CardHeader
            title={t("dashboard.analyses_30d")}
            subtitle={t("dashboard.analyses_30d_hint")}
            icon={<Activity size={18} />}
          />
          <CardBody>
            {loading ? <SkeletonChart /> : <AnalysesOverTime data={series} locale={locale} />}
          </CardBody>
        </Card>

        <Card>
          <CardHeader
            title={t("dashboard.status_breakdown")}
            icon={<Sprout size={18} />}
          />
          <CardBody>
            <StatusBreakdown slices={slices} />
          </CardBody>
        </Card>
      </div>

      <Card>
        <CardHeader title={t("dashboard.latest")} />
        {loading ? (
          <ul className="divide-y divide-slate-100">
            {Array.from({ length: 3 }).map((_, i) => (
              <li key={i}>
                <SkeletonRow />
              </li>
            ))}
          </ul>
        ) : items.length === 0 ? (
          <CardBody>
            <div className="py-8 text-center">
              <Sprout className="mx-auto text-slate-300" size={36} aria-hidden />
              <p className="text-sm text-slate-500 mt-3">{t("dashboard.empty")}</p>
              <Link
                to="/upload"
                className="inline-flex items-center gap-1.5 mt-4 rounded-md bg-brand-600 hover:bg-brand-700 text-white px-4 py-2 text-sm font-semibold"
              >
                <Plus size={16} aria-hidden />
                {t("dashboard.new_analysis")}
              </Link>
            </div>
          </CardBody>
        ) : (
          <ul className="divide-y divide-slate-100">
            {items.slice(0, 5).map((a) => (
              <li key={a.id}>
                <Link
                  to={`/analyses/${a.id}`}
                  className="flex items-center justify-between p-4 hover:bg-slate-50 transition-colors"
                >
                  <div className="min-w-0">
                    <div className="font-medium text-slate-900 truncate">
                      {a.source_name}
                    </div>
                    <div className="text-xs text-slate-500 mt-0.5">
                      {new Date(a.created_at).toLocaleString(locale)}
                    </div>
                  </div>
                  <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-slate-50 text-slate-700 border border-slate-200">
                    <span
                      className={`inline-block w-1.5 h-1.5 rounded-full ${
                        STATUS_DOTS[a.status] ?? "bg-slate-400"
                      }`}
                    />
                    {t(`status.${a.status}`)}
                  </span>
                </Link>
              </li>
            ))}
          </ul>
        )}
      </Card>
    </div>
  );
}
