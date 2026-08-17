import { Area, AreaChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { AnomalyTrendPoint } from "../../api/types";
import { formatTimestamp } from "../../utils/format";

export default function AnomalyTrendChart({
  data,
  threshold,
}: {
  data: AnomalyTrendPoint[];
  threshold?: number | null;
}) {
  const chartData = data.map((d) => ({
    ...d,
    label: new Date(d.timestamp).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" }),
  }));

  return (
    <ResponsiveContainer width="100%" height={220}>
      <AreaChart data={chartData} margin={{ top: 8, right: 12, bottom: 0, left: -12 }}>
        <defs>
          <linearGradient id="anomalyFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--color-accent)" stopOpacity={0.28} />
            <stop offset="100%" stopColor="var(--color-accent)" stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <CartesianGrid vertical={false} stroke="var(--color-border)" strokeDasharray="3 4" />
        <XAxis
          dataKey="label"
          tick={{ fontSize: 10, fontFamily: "var(--font-mono)", fill: "var(--color-ink-muted)" }}
          axisLine={{ stroke: "var(--color-border)" }}
          tickLine={false}
          minTickGap={32}
        />
        <YAxis
          tick={{ fontSize: 10, fontFamily: "var(--font-mono)", fill: "var(--color-ink-muted)" }}
          axisLine={false}
          tickLine={false}
          width={44}
        />
        {threshold != null && (
          <ReferenceLine
            y={threshold}
            stroke="var(--color-critical)"
            strokeDasharray="4 4"
            strokeWidth={1}
            label={{
              value: "threshold",
              position: "insideTopRight",
              fontSize: 10,
              fill: "var(--color-critical)",
              fontFamily: "var(--font-mono)",
            }}
          />
        )}
        <Tooltip
          contentStyle={{
            background: "var(--color-canvas-raised)",
            border: "1px solid var(--color-border)",
            borderRadius: 8,
            fontSize: 12,
            fontFamily: "var(--font-mono)",
          }}
          labelFormatter={(_, payload) => (payload?.[0] ? formatTimestamp(payload[0].payload.timestamp) : "")}
          formatter={(value) => [typeof value === "number" ? value.toFixed(4) : String(value), "anomaly score"]}
        />
        <Area
          type="monotone"
          dataKey="anomaly_score"
          stroke="var(--color-accent)"
          strokeWidth={1.75}
          fill="url(#anomalyFill)"
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
