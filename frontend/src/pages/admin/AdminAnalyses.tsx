import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { AdminAnalysisRow, deleteAnalysis, listAnalyses } from "@/api/admin";

const STATUSES = ["all", "queued", "processing", "completed", "failed"] as const;

const statusColor: Record<string, string> = {
  queued: "bg-slate-200 text-slate-700",
  processing: "bg-amber-100 text-amber-800",
  completed: "bg-emerald-100 text-emerald-800",
  failed: "bg-red-100 text-red-700",
};

export default function AdminAnalysesPage() {
  const { t, i18n } = useTranslation();
  const [rows, setRows] = useState<AdminAnalysisRow[]>([]);
  const [filter, setFilter] = useState<(typeof STATUSES)[number]>("all");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const locale = i18n.language.split("-")[0];

  async function refresh() {
    setLoading(true);
    setError(null);
    try {
      setRows(await listAnalyses(filter === "all" ? undefined : { status: filter }));
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response
        ?.data?.detail;
      setError(typeof detail === "string" ? detail : t("admin.analyses.load_failed"));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
  }, [filter]);

  async function onDelete(row: AdminAnalysisRow) {
    if (!confirm(t("admin.analyses.confirm_delete", { name: row.source_name }) ?? "")) return;
    try {
      await deleteAnalysis(row.id);
      setRows((rs) => rs.filter((r) => r.id !== row.id));
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response
        ?.data?.detail;
      setError(typeof detail === "string" ? detail : t("admin.analyses.delete_failed"));
    }
  }

  return (
    <div className="space-y-4">
      <header>
        <h1 className="text-2xl font-bold">{t("admin.nav.analyses")}</h1>
      </header>
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm text-slate-500 me-2">{t("reports.filter")}:</span>
        {STATUSES.map((s) => (
          <button
            key={s}
            onClick={() => setFilter(s)}
            className={`px-3 py-1.5 rounded-full text-xs font-medium transition ${
              filter === s
                ? "bg-brand-600 text-white"
                : "bg-slate-100 text-slate-700 hover:bg-slate-200"
            }`}
          >
            {t(`status.${s}`)}
          </button>
        ))}
      </div>

      {error && (
        <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-md p-2">
          {error}
        </div>
      )}

      {loading ? (
        <p className="text-sm text-slate-500">{t("common.loading")}</p>
      ) : rows.length === 0 ? (
        <p className="text-sm text-slate-500">{t("admin.analyses.empty")}</p>
      ) : (
        <ul className="bg-white rounded-lg shadow-sm divide-y divide-slate-200">
          {rows.map((row) => (
            <li
              key={row.id}
              className="p-3 flex flex-col md:flex-row md:items-center gap-3"
            >
              <div className="min-w-0 md:flex-1">
                <Link
                  to={`/analyses/${row.id}`}
                  state={{ from: "analyses" }}
                  className="font-medium truncate hover:text-brand-700 hover:underline block"
                >
                  {row.source_name}
                </Link>
                <div className="text-xs text-slate-500 truncate">
                  {row.user_display_name} • {row.user_email}
                </div>
                <div className="text-[10px] text-slate-400 mt-0.5">
                  {new Date(row.created_at).toLocaleString(locale)}
                </div>
                {row.error && (
                  <div className="text-xs text-red-600 mt-1 line-clamp-2">{row.error}</div>
                )}
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <span
                  className={`px-2 py-1 rounded-md text-xs font-medium ${
                    statusColor[row.status] ?? "bg-slate-100"
                  }`}
                >
                  {t(`status.${row.status}`)}
                </span>
                <button
                  onClick={() => onDelete(row)}
                  className="text-xs px-2 py-1 rounded-md text-red-600 hover:bg-red-50"
                >
                  {t("common.delete")}
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
