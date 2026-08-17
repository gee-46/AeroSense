import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { FaultDistribution } from "../../api/types";
import { formatClassLabel } from "../../utils/format";

const BAR_COLORS = ["#8993b3", "#6a76a0", "#3e4c7a", "#2c3860"];

export default function FaultDistributionChart({ data }: { data: FaultDistribution }) {
  const chartData = Object.entries(data)
    .map(([key, value]) => ({ key, label: formatClassLabel(key), value }))
    .sort((a, b) => b.value - a.value);

  return (
    <ResponsiveContainer width="100%" height={Math.max(160, chartData.length * 46)}>
      <BarChart data={chartData} layout="vertical" margin={{ top: 0, right: 24, bottom: 0, left: 8 }}>
        <CartesianGrid horizontal={false} stroke="var(--color-border)" strokeDasharray="3 4" />
        <XAxis
          type="number"
          allowDecimals={false}
          tick={{ fontSize: 10, fontFamily: "var(--font-mono)", fill: "var(--color-ink-muted)" }}
          axisLine={{ stroke: "var(--color-border)" }}
          tickLine={false}
        />
        <YAxis
          type="category"
          dataKey="label"
          width={140}
          tick={{ fontSize: 12, fontFamily: "var(--font-body)", fill: "var(--color-ink)" }}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip
          cursor={{ fill: "var(--color-canvas-sunken)" }}
          contentStyle={{
            background: "var(--color-canvas-raised)",
            border: "1px solid var(--color-border)",
            borderRadius: 8,
            fontSize: 12,
          }}
        />
        <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={18}>
          {chartData.map((entry, i) => (
            <Cell key={entry.key} fill={BAR_COLORS[i % BAR_COLORS.length]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
