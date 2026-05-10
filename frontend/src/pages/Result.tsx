import { useEffect, useState } from "react";
import { Link, useLocation, useParams } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { ArrowLeft, Download } from "lucide-react";
import { AnalysisDetail, api } from "@/api/client";
import { useAnalysisEvents } from "@/hooks/useSSE";
import { pdfUrl } from "@/api/reports";
import Timeline from "@/components/Timeline";

const STATUS_DOT: Record<string, string> = {
  queued: "bg-slate-400",
  processing: "bg-amber-500",
  completed: "bg-emerald-500",
  failed: "bg-red-500",
};

// Pre-enum results stored free-text trajectory strings produced by the
// heuristic. Map them onto the new canonical codes so legacy reports still
// pick up the localized chip; otherwise fall back to the raw text.
const TRAJECTORY_ALIAS: Record<string, string> = {
  "DECLINING (significant drop across season)": "STRONG_DECLINE",
  "SLIGHTLY DECLINING": "DECLINE",
  STABLE: "STABLE",
  GROWING: "GROWTH",
  "STRONG GROWTH": "STRONG_GROWTH",
};

function trajectoryCode(raw: string): string {
  return TRAJECTORY_ALIAS[(raw ?? "").toUpperCase()] ?? raw;
}

export default function ResultPage() {
  const { t, i18n } = useTranslation();
  const { id } = useParams<{ id: string }>();
  const location = useLocation();
  const from = (location.state as { from?: string } | null)?.from;
  const backTo =
    from === "reports"
      ? "/reports"
      : from === "analyses"
      ? "/analyses"
      : "/dashboard";
  const backLabel =
    from === "reports"
      ? t("nav.reports")
      : from === "analyses"
      ? t("nav.analyses")
      : t("nav.dashboard");
  const analysisId = id ? Number(id) : null;
  const [detail, setDetail] = useState<AnalysisDetail | null>(null);

  async function refresh() {
    if (!analysisId) return;
    const { data } = await api.get<AnalysisDetail>(`/analyses/${analysisId}`);
    setDetail(data);
  }

  useEffect(() => {
    refresh();
  }, [analysisId]);

  const inFlight = detail?.status === "queued" || detail?.status === "processing";
  const { stage } = useAnalysisEvents(analysisId, !!inFlight);

  useEffect(() => {
    if (stage === "done" || stage === "failed") refresh();
  }, [stage]);

  if (!detail) return <p className="text-sm text-slate-500">{t("common.loading")}</p>;

  const lang = i18n.language.split("-")[0];

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div className="min-w-0">
          <Link
            to={backTo}
            className="inline-flex items-center gap-1 text-sm text-brand-700 hover:underline"
          >
            <ArrowLeft size={14} aria-hidden /> {backLabel}
          </Link>
          <h1 className="text-2xl font-bold mt-1 break-words text-slate-900">
            {detail.source_name}
          </h1>
        </div>
        <div className="flex flex-wrap items-center gap-2 md:gap-3">
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-slate-50 text-slate-700 border border-slate-200">
            <span
              className={`inline-block w-1.5 h-1.5 rounded-full ${
                STATUS_DOT[detail.status] ?? "bg-slate-400"
              }`}
            />
            {t(`status.${detail.status}`)}
          </span>
          {detail.result && (
            <a
              href={pdfUrl(detail.id, lang)}
              className="inline-flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-md bg-brand-600 hover:bg-brand-700 text-white shadow-sm"
            >
              <Download size={14} aria-hidden />
              {t("common.download_pdf")}
            </a>
          )}
        </div>
      </div>

      {(inFlight || detail.status === "failed") && (
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
          <h2 className="font-semibold mb-4 text-slate-900">
            {detail.status === "failed"
              ? t("result.failed")
              : t("result.processing")}
          </h2>
          <Timeline
            stage={detail.status === "failed" ? "failed" : stage}
            labels={{
              queued: t("stages.queued"),
              loading: t("stages.loading"),
              inferring: t("stages.inferring"),
              rendering: t("stages.rendering"),
              done: t("stages.done"),
            }}
            failedMessage={detail.error}
          />
        </div>
      )}

      {detail.result && (
        <>
          <section className="grid gap-4 md:grid-cols-2">
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
              <h2 className="font-semibold mb-3 text-slate-900">
                {t("result.classification_map")}
              </h2>
              <img
                src={detail.result.map_url}
                alt={t("result.classification_map") ?? ""}
                className="mx-auto max-h-[420px] w-auto max-w-full rounded-md border border-slate-200 bg-slate-50"
              />
              {detail.result.legend &&
                Object.keys(detail.result.legend).length > 0 && (
                  <div className="mt-4">
                    <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500 mb-2">
                      {t("result.legend")}
                    </h3>
                    <ul className="grid grid-cols-1 sm:grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
                      {Object.entries(detail.result.legend)
                        .sort((a, b) => {
                          const da =
                            detail.result?.class_distribution[a[0]] ?? 0;
                          const db =
                            detail.result?.class_distribution[b[0]] ?? 0;
                          return db - da;
                        })
                        .map(([cls, color]) => (
                          <li
                            key={cls}
                            className="flex items-center gap-2 min-w-0"
                          >
                            <span
                              className="inline-block w-3 h-3 rounded-sm border border-slate-300 shrink-0"
                              style={{ background: color }}
                              aria-hidden
                            />
                            <span className="text-slate-700 truncate">
                              {cls.replace(/_/g, " ")}
                            </span>
                          </li>
                        ))}
                    </ul>
                  </div>
                )}
            </div>

            {detail.result.confidence_url && (
              <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
                <h2 className="font-semibold mb-3 text-slate-900">
                  {t("result.confidence_map")}
                </h2>
                <img
                  src={detail.result.confidence_url}
                  alt={t("result.confidence_map") ?? ""}
                  className="mx-auto max-h-[420px] w-auto max-w-full rounded-md border border-slate-200 bg-slate-50"
                />
                <div className="mt-4">
                  <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500 mb-2">
                    {t("result.legend")}
                  </h3>
                  <div
                    className="h-3 w-full rounded-full border border-slate-200"
                    style={{
                      background:
                        "linear-gradient(to right, #d73027, #fdae61, #fee08b, #d9ef8b, #1a9850)",
                    }}
                    aria-hidden
                  />
                  <div className="flex justify-between text-[10px] text-slate-500 mt-1 tabular-nums">
                    <span>0.0 — {t("result.confidence_low")}</span>
                    <span>0.5</span>
                    <span>1.0 — {t("result.confidence_high")}</span>
                  </div>
                </div>
              </div>
            )}
          </section>

          <section className="bg-white rounded-lg shadow-sm p-5 space-y-3">
            <h2 className="font-semibold">{t("result.summary")}</h2>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
              <Field
                label={t("result.predicted_crop")}
                value={detail.result.predicted_crop.replace(/_/g, " ")}
              />
              <Field
                label={t("result.health")}
                value={t(`result.health_codes.${detail.result.health_status}`, {
                  defaultValue: detail.result.health_status,
                })}
              />
              <Field
                label={t("result.trajectory")}
                value={t(
                  `result.trajectory_codes.${trajectoryCode(detail.result.trajectory)}`,
                  { defaultValue: detail.result.trajectory },
                )}
              />
              <Field
                label={t("result.confidence")}
                value={(detail.result.mean_confidence * 100).toFixed(1) + "%"}
              />
              <Field label={t("result.ndvi")} value={detail.result.mean_ndvi.toFixed(3)} />
              <Field label={t("result.ndre")} value={detail.result.mean_ndre.toFixed(3)} />
              <Field label={t("result.savi")} value={detail.result.mean_savi.toFixed(3)} />
              <Field
                label={t("result.temporal_trend")}
                value={detail.result.temporal_trend.toFixed(3)}
              />
            </div>
            <div className="rounded-md bg-amber-50 border border-amber-200 p-3 text-sm text-amber-900">
              <strong>{t("result.advice")}:</strong> {detail.result.advice}
            </div>
          </section>

          {detail.result.farmer_notes && (
            <section className="bg-white rounded-lg shadow-sm p-5">
              <div className="flex items-baseline justify-between gap-3 mb-2">
                <h2 className="font-semibold">{t("reports.farmer_notes")}</h2>
                {detail.result.observed_at && (
                  <span className="text-xs text-slate-500">
                    {t("reports.observed_on")}:{" "}
                    {new Date(detail.result.observed_at).toLocaleDateString(lang)}
                  </span>
                )}
              </div>
              <p className="text-sm text-slate-700 whitespace-pre-wrap">
                {detail.result.farmer_notes}
              </p>
            </section>
          )}

          <section className="bg-white rounded-lg shadow-sm p-5">
            <h2 className="font-semibold mb-3">{t("result.class_distribution")}</h2>
            <ul className="text-sm divide-y divide-slate-200">
              {Object.entries(detail.result.class_distribution).map(([cls, frac]) => (
                <li key={cls} className="flex items-center justify-between py-2">
                  <span className="truncate">{cls.replace(/_/g, " ")}</span>
                  <span className="font-mono">{(frac * 100).toFixed(1)}%</span>
                </li>
              ))}
            </ul>
          </section>
        </>
      )}
    </div>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs text-slate-500">{label}</div>
      <div className="font-medium">{value}</div>
    </div>
  );
}
