import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { listReports, pdfUrl, ReportRow, updateNotes } from "@/api/reports";

const STATUSES = ["all", "queued", "processing", "completed", "failed"] as const;

export default function ReportsPage() {
  const { t, i18n } = useTranslation();
  const [filter, setFilter] = useState<(typeof STATUSES)[number]>("all");
  const [rows, setRows] = useState<ReportRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [openId, setOpenId] = useState<number | null>(null);

  async function refresh() {
    setLoading(true);
    try {
      const data = await listReports(filter === "all" ? undefined : { status: filter });
      setRows(data);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
  }, [filter]);

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold">{t("reports.title")}</h1>
        <p className="text-sm text-slate-600 mt-1">{t("reports.subtitle")}</p>
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

      {loading ? (
        <p className="text-sm text-slate-500">{t("common.loading")}</p>
      ) : rows.length === 0 ? (
        <p className="text-sm text-slate-500">{t("reports.empty")}</p>
      ) : (
        <ul className="space-y-3">
          {rows.map((row) => (
            <ReportCard
              key={row.id}
              row={row}
              language={i18n.language.split("-")[0]}
              isOpen={openId === row.id}
              onToggle={() => setOpenId(openId === row.id ? null : row.id)}
              onSaved={refresh}
            />
          ))}
        </ul>
      )}
    </div>
  );
}

function ReportCard({
  row,
  language,
  isOpen,
  onToggle,
  onSaved,
}: {
  row: ReportRow;
  language: string;
  isOpen: boolean;
  onToggle: () => void;
  onSaved: () => void;
}) {
  const { t } = useTranslation();
  const [notes, setNotes] = useState(row.farmer_notes ?? "");
  const [observedAt, setObservedAt] = useState(
    row.observed_at ? row.observed_at.slice(0, 10) : ""
  );
  const [saving, setSaving] = useState(false);
  const [feedback, setFeedback] = useState<{ kind: "ok" | "err"; msg: string } | null>(null);

  useEffect(() => {
    setNotes(row.farmer_notes ?? "");
    setObservedAt(row.observed_at ? row.observed_at.slice(0, 10) : "");
  }, [row.id, row.farmer_notes, row.observed_at]);

  async function save() {
    setSaving(true);
    setFeedback(null);
    try {
      await updateNotes(row.id, {
        farmer_notes: notes.trim() ? notes.trim() : null,
        observed_at: observedAt ? new Date(observedAt).toISOString() : null,
      });
      setFeedback({ kind: "ok", msg: t("reports.notes_saved") });
      onSaved();
    } catch {
      setFeedback({ kind: "err", msg: t("reports.save_failed") });
    } finally {
      setSaving(false);
    }
  }

  const statusColor: Record<string, string> = {
    queued: "bg-slate-200 text-slate-700",
    processing: "bg-amber-100 text-amber-800",
    completed: "bg-emerald-100 text-emerald-800",
    failed: "bg-red-100 text-red-700",
  };

  return (
    <li className="bg-white rounded-lg shadow-sm">
      <div className="p-4 flex flex-wrap items-center gap-3 justify-between">
        <div className="min-w-0 flex-1">
          <div className="font-medium truncate">{row.source_name}</div>
          <div className="text-xs text-slate-500 mt-0.5">
            {new Date(row.created_at).toLocaleString(language)}
          </div>
          {row.has_result && (
            <div className="mt-2 flex flex-wrap gap-3 text-xs text-slate-600">
              <span>
                {t("reports.row_predicted")}:{" "}
                <strong>{row.predicted_crop?.replace(/_/g, " ") ?? "—"}</strong>
              </span>
              <span>
                {t("reports.row_health")}: <strong>{row.health_status ?? "—"}</strong>
              </span>
            </div>
          )}
        </div>
        <span
          className={`px-2 py-1 rounded-md text-xs font-medium ${
            statusColor[row.status] ?? "bg-slate-100"
          }`}
        >
          {t(`status.${row.status}`)}
        </span>
        <div className="flex items-center gap-2">
          {row.has_result && (
            <>
              <button
                onClick={onToggle}
                className="text-xs px-3 py-1.5 rounded-md border border-slate-300 hover:bg-slate-50"
              >
                {isOpen ? t("reports.hide_notes") : t("reports.edit_notes")}
              </button>
              <a
                href={pdfUrl(row.id, language)}
                className="text-xs px-3 py-1.5 rounded-md bg-brand-600 hover:bg-brand-700 text-white"
              >
                {t("common.download_pdf")}
              </a>
            </>
          )}
        </div>
      </div>
      {isOpen && row.has_result && (
        <div className="border-t border-slate-200 p-4 space-y-3 bg-slate-50">
          <label className="block text-sm font-medium">
            {t("reports.farmer_notes")}
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              rows={3}
              placeholder={t("reports.notes_placeholder") ?? ""}
              className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-brand-500 focus:ring-brand-500 bg-white"
            />
          </label>
          <label className="block text-sm">
            {t("reports.observed_on")}
            <input
              type="date"
              value={observedAt}
              onChange={(e) => setObservedAt(e.target.value)}
              className="mt-1 rounded-md border border-slate-300 px-3 py-2 text-sm bg-white"
            />
          </label>
          <div className="flex items-center gap-3">
            <button
              onClick={save}
              disabled={saving}
              className="rounded-md bg-brand-600 hover:bg-brand-700 text-white px-4 py-2 text-sm font-medium disabled:opacity-50"
            >
              {saving ? t("common.loading") : t("common.save")}
            </button>
            {feedback && (
              <span
                className={`text-sm ${
                  feedback.kind === "ok" ? "text-emerald-700" : "text-red-600"
                }`}
              >
                {feedback.msg}
              </span>
            )}
          </div>
        </div>
      )}
    </li>
  );
}
