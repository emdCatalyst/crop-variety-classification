import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export type DailyPoint = {
  date: string;
  count: number;
  completed?: number;
  failed?: number;
};

export default function AnalysesOverTime({
  data,
  locale,
}: {
  data: DailyPoint[];
  locale: string;
}) {
  const fmt = new Intl.DateTimeFormat(locale, { month: "short", day: "numeric" });
  const ticks = data.map((d) => ({ ...d, label: fmt.format(new Date(d.date)) }));

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={ticks} margin={{ top: 10, right: 12, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="barFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#3f8a3a" stopOpacity={0.9} />
            <stop offset="95%" stopColor="#3f8a3a" stopOpacity={0.5} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
        <XAxis
          dataKey="label"
          tick={{ fontSize: 10, fill: "#64748b" }}
          interval="preserveStartEnd"
          minTickGap={20}
          axisLine={{ stroke: "#e2e8f0" }}
          tickLine={false}
        />
        <YAxis
          allowDecimals={false}
          tick={{ fontSize: 10, fill: "#64748b" }}
          axisLine={false}
          tickLine={false}
          width={28}
        />
        <Tooltip
          cursor={{ fill: "#f1f5f9" }}
          contentStyle={{
            background: "white",
            border: "1px solid #e2e8f0",
            borderRadius: 8,
            fontSize: 12,
          }}
          labelStyle={{ color: "#0f172a", fontWeight: 600 }}
        />
        <Bar dataKey="count" fill="url(#barFill)" radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
