import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { AnalysisDetail, api } from "@/api/client";
import { useAnalysisEvents } from "@/hooks/useSSE";

export default function ResultPage() {
  const { id } = useParams<{ id: string }>();
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
  const { events, stage } = useAnalysisEvents(analysisId, !!inFlight);

  useEffect(() => {
    if (stage === "done" || stage === "failed") refresh();
  }, [stage]);

  if (!detail) return <p className="text-sm text-slate-500">Loading…</p>;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <Link to="/dashboard" className="text-sm text-brand-700 hover:underline">
            ← Dashboard
          </Link>
          <h1 className="text-2xl font-bold mt-1">{detail.source_name}</h1>
        </div>
        <span className="px-3 py-1 rounded-md text-xs font-medium bg-slate-200 text-slate-700">
          {detail.status}
        </span>
      </div>

      {inFlight && (
        <div className="bg-white rounded-lg shadow-sm p-5">
          <h2 className="font-semibold mb-2">Processing…</h2>
          <p className="text-sm text-slate-600 mb-3">
            Current stage: <strong>{stage}</strong>
          </p>
          <ol className="text-sm text-slate-700 space-y-1 max-h-48 overflow-y-auto">
            {events.map((e, i) => (
              <li key={i}>
                <span className="text-slate-400 me-2">[{e.stage}]</span>
                {e.message}
              </li>
            ))}
          </ol>
        </div>
      )}

      {detail.status === "failed" && (
        <div className="rounded-md bg-red-50 border border-red-200 text-red-700 p-4 text-sm">
          {detail.error ?? "Analysis failed"}
        </div>
      )}

      {detail.result && (
        <>
          <section className="bg-white rounded-lg shadow-sm p-5">
            <h2 className="font-semibold mb-3">Classification map</h2>
            <img
              src={detail.result.map_url}
              alt="Classification map"
              className="w-full rounded-md border border-slate-200"
            />
          </section>

          <section className="bg-white rounded-lg shadow-sm p-5 space-y-3">
            <h2 className="font-semibold">Summary</h2>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
              <Field label="Predicted crop" value={detail.result.predicted_crop} />
              <Field label="Health" value={detail.result.health_status} />
              <Field label="Trajectory" value={detail.result.trajectory} />
              <Field
                label="Confidence"
                value={(detail.result.mean_confidence * 100).toFixed(1) + "%"}
              />
              <Field label="NDVI" value={detail.result.mean_ndvi.toFixed(3)} />
              <Field label="NDRE" value={detail.result.mean_ndre.toFixed(3)} />
              <Field label="SAVI" value={detail.result.mean_savi.toFixed(3)} />
              <Field
                label="Temporal trend"
                value={detail.result.temporal_trend.toFixed(3)}
              />
            </div>
            <div className="rounded-md bg-amber-50 border border-amber-200 p-3 text-sm text-amber-900">
              <strong>Advice:</strong> {detail.result.advice}
            </div>
          </section>

          <section className="bg-white rounded-lg shadow-sm p-5">
            <h2 className="font-semibold mb-3">Class distribution</h2>
            <ul className="text-sm divide-y divide-slate-200">
              {Object.entries(detail.result.class_distribution).map(([cls, frac]) => (
                <li key={cls} className="flex items-center justify-between py-2">
                  <span className="truncate">{cls}</span>
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
