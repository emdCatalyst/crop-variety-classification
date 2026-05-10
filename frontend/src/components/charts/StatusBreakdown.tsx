import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";

const COLORS: Record<string, string> = {
  queued: "#94a3b8",
  processing: "#f59e0b",
  completed: "#10b981",
  failed: "#ef4444",
};

export type StatusSlice = { key: string; label: string; value: number };

export default function StatusBreakdown({ slices }: { slices: StatusSlice[] }) {
  const total = slices.reduce((s, x) => s + x.value, 0);
  const data = slices.filter((s) => s.value > 0);

  if (total === 0) {
    return (
      <div className="h-[220px] flex items-center justify-center text-sm text-slate-400">
        —
      </div>
    );
  }

  return (
    <div className="flex items-center gap-4">
      <div className="w-[160px] h-[160px] shrink-0">
        <ResponsiveContainer>
          <PieChart>
            <Pie
              data={data}
              dataKey="value"
              innerRadius={48}
              outerRadius={72}
              paddingAngle={2}
              stroke="white"
              strokeWidth={2}
            >
              {data.map((d) => (
                <Cell key={d.key} fill={COLORS[d.key] ?? "#cbd5e1"} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{
                background: "white",
                border: "1px solid #e2e8f0",
                borderRadius: 8,
                fontSize: 12,
              }}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>
      <ul className="flex-1 min-w-0 space-y-1.5 text-sm">
        {slices.map((s) => {
          const pct = total > 0 ? Math.round((s.value / total) * 100) : 0;
          return (
            <li key={s.key} className="flex items-center gap-2">
              <span
                className="inline-block w-2.5 h-2.5 rounded-full shrink-0"
                style={{ background: COLORS[s.key] ?? "#cbd5e1" }}
              />
              <span className="text-slate-700 flex-1 truncate">{s.label}</span>
              <span className="text-slate-400 text-xs tabular-nums">
                {s.value} · {pct}%
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
