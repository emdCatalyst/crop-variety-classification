import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Legend,
} from "recharts";

export type AdminPoint = { date: string; analyses: number; new_users: number };

export default function AdminTimeseries({
  data,
  locale,
  labels,
}: {
  data: AdminPoint[];
  locale: string;
  labels: { analyses: string; new_users: string };
}) {
  const fmt = new Intl.DateTimeFormat(locale, { month: "short", day: "numeric" });
  const points = data.map((d) => ({ ...d, label: fmt.format(new Date(d.date)) }));

  return (
    <ResponsiveContainer width="100%" height={240}>
      <LineChart data={points} margin={{ top: 10, right: 12, left: 0, bottom: 0 }}>
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
          contentStyle={{
            background: "white",
            border: "1px solid #e2e8f0",
            borderRadius: 8,
            fontSize: 12,
          }}
          labelStyle={{ color: "#0f172a", fontWeight: 600 }}
        />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Line
          type="monotone"
          dataKey="analyses"
          stroke="#3f8a3a"
          strokeWidth={2}
          dot={false}
          name={labels.analyses}
        />
        <Line
          type="monotone"
          dataKey="new_users"
          stroke="#0ea5e9"
          strokeWidth={2}
          dot={false}
          name={labels.new_users}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
