import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Analysis, api } from "@/api/client";

const statusBadge: Record<string, string> = {
  queued: "bg-slate-200 text-slate-700",
  processing: "bg-amber-100 text-amber-800",
  completed: "bg-emerald-100 text-emerald-800",
  failed: "bg-red-100 text-red-700",
};

export default function DashboardPage() {
  const [items, setItems] = useState<Analysis[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .get<Analysis[]>("/analyses")
      .then((r) => setItems(r.data))
      .finally(() => setLoading(false));
  }, []);

  const total = items.length;
  const processing = items.filter((a) => a.status === "queued" || a.status === "processing").length;
  const completed = items.filter((a) => a.status === "completed").length;

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between">
        <h1 className="text-2xl font-bold">Dashboard</h1>
        <Link
          to="/upload"
          className="rounded-md bg-brand-600 hover:bg-brand-700 text-white px-4 py-2 text-sm font-medium"
        >
          New analysis
        </Link>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Stat label="Total analyses" value={total} />
        <Stat label="Processing" value={processing} />
        <Stat label="Completed" value={completed} />
      </div>

      <section>
        <h2 className="text-lg font-semibold mb-2">Latest analyses</h2>
        {loading ? (
          <p className="text-sm text-slate-500">Loading…</p>
        ) : items.length === 0 ? (
          <p className="text-sm text-slate-500">No analyses yet. Start your first one.</p>
        ) : (
          <ul className="divide-y divide-slate-200 bg-white rounded-lg shadow-sm">
            {items.slice(0, 10).map((a) => (
              <li key={a.id} className="p-4 flex items-center justify-between">
                <Link to={`/analyses/${a.id}`} className="flex-1">
                  <div className="font-medium">{a.source_name}</div>
                  <div className="text-xs text-slate-500">
                    {new Date(a.created_at).toLocaleString()}
                  </div>
                </Link>
                <span
                  className={`px-2 py-1 rounded-md text-xs font-medium ${
                    statusBadge[a.status] ?? "bg-slate-100"
                  }`}
                >
                  {a.status}
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="bg-white rounded-lg shadow-sm p-4">
      <div className="text-sm text-slate-500">{label}</div>
      <div className="text-3xl font-bold mt-1">{value}</div>
    </div>
  );
}
